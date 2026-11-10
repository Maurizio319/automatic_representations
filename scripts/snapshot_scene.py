"""PyMOL scene of one interaction group: loading, styling, camera, rendering,
and the JSON metadata embedded in each image.

Shared by medoid_snapshots.py and recover_view.py, so a recovered view is the
same scene. Every setting comes from the config's `style` / `image` sections.
"""
import json
import os
import struct
import zlib

from pymol import cmd, util

import labels

OBJECT = "sys"           # PyMOL object holding the structure
MEASURE_PREFIX = "ct_"   # prefix of every dashed-line object
GROUP_SELECTION = "group_res"
METADATA_KEY = "medoid_snapshot"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_topology(topology):
    """Load the topology as OBJECT, in file order (= MDAnalysis order)."""
    cmd.delete("all")
    cmd.set("retain_order", 1)
    cmd.load(topology, OBJECT)


def load_frame_from_trajectory(topology, trajectory, frame):
    """Topology + one 0-based frame, with PyMOL alone (no MDAnalysis)."""
    load_topology(topology)
    cmd.load_traj(trajectory, OBJECT, state=1, start=frame + 1, stop=frame + 1)


def set_coordinates(positions):
    cmd.load_coords(positions, OBJECT, state=1)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def residue_selection(label):
    return labels.residue_selection(label, OBJECT)


def apply_render_settings(style, image):
    cmd.bg_color(style["background"])
    cmd.viewport(image["width"], image["height"])
    for name, value in style["pymol_settings"].items():
        cmd.set(name, value)


def _closest_heavy_atoms(sel_a, sel_b):
    """(index_a, index_b) of the closest heavy-atom pair."""
    atoms = {"a": [], "b": []}
    for key, sel in (("a", sel_a), ("b", sel_b)):
        cmd.iterate_state(1, f"{sel} and not hydro", "atoms[key].append((index, x, y, z))",
                          space={"atoms": atoms, "key": key})
    best = None
    for ia, xa, ya, za in atoms["a"]:
        for ib, xb, yb, zb in atoms["b"]:
            d2 = (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2
            if best is None or d2 < best[0]:
                best = (d2, ia, ib)
    return best[1], best[2]


def _polar_heavy_pairs(name, sel):
    """Atom pairs of polar-contact object `name`, H ends replaced by their heavy atom.

    PyMOL has no public call for a distance object's end points, so they are
    read from its session data and matched back to the atoms of `sel`."""
    try:
        ends = cmd.get_session(name)["names"][0][5][2][0][1]
    except (IndexError, KeyError, TypeError) as err:
        raise RuntimeError("This PyMOL version stores distance objects differently: "
                           "update _polar_heavy_pairs in snapshot_scene.py") from err
    atoms = []
    cmd.iterate_state(1, sel, "atoms.append((index, elem, x, y, z))", space={"atoms": atoms})

    def atom_at(x, y, z):
        index, elem, *_ = min(atoms, key=lambda a: (a[2] - x) ** 2 + (a[3] - y) ** 2 + (a[4] - z) ** 2)
        if elem == "H":
            heavy = []
            cmd.iterate(f"(neighbor ({OBJECT} and index {index})) and not hydro", "heavy.append(index)",
                        space={"heavy": heavy})
            index = heavy[0]
        return index

    pairs = []
    for i in range(0, len(ends), 6):
        pair = (atom_at(*ends[i:i + 3]), atom_at(*ends[i + 3:i + 6]))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _draw_contact(name, lig, prot, style):
    """Dashed line between two residues: PyMOL polar contacts if any, else the
    closest heavy atoms. Returns "polar" or "closest"."""
    found = cmd.distance(name, residue_selection(lig), residue_selection(prot), mode=2)
    if found != 0 and style["hide_hydrogens"]:
        # hydrogens are hidden: redraw between heavy atoms
        heavy_pairs = _polar_heavy_pairs(name, f"{residue_selection(lig)} or {residue_selection(prot)}")
        cmd.delete(name)
        for ia, ib in heavy_pairs:
            cmd.distance(name, f"{OBJECT} and index {ia}", f"{OBJECT} and index {ib}", mode=0)
    if found != 0:
        return "polar"
    cmd.delete(name)
    ia, ib = _closest_heavy_atoms(residue_selection(lig), residue_selection(prot))
    cmd.distance(name, f"{OBJECT} and index {ia}", f"{OBJECT} and index {ib}", mode=0)
    return "closest"


def build_scene(pairs, active, style, status=None):
    """Style OBJECT for one group of (ligand, protein) label pairs.

    active: per pair, in contact in this frame. Dashes for active pairs (all
    if style["show_inactive_contacts"]), colored polar/closest.
    status: reference mode, per pair "kept", "gained", "lost" or None; dashes
    for every non-None pair, colored by status.

    Returns [(ligand, protein, kind)] of the dashes drawn.
    """
    cmd.delete(f"{MEASURE_PREFIX}*")
    cmd.label(OBJECT, "")
    cmd.hide("everything", OBJECT)

    # context
    protein, nucleic = f"{OBJECT} and polymer.protein", f"{OBJECT} and polymer.nucleic"
    cmd.dss(OBJECT)
    for rep in style["context_representations"]:
        cmd.show(rep, OBJECT)
    cmd.set("cartoon_transparency", style["cartoon_transparency"], OBJECT)
    cmd.color(style["cartoon_color_protein"], protein)
    cmd.color(style["cartoon_color_nucleic"], nucleic)

    # group residues: carbons by role, other atoms by element
    ligands = sorted({a for a, _ in pairs})
    proteins = sorted({b for _, b in pairs})
    lig_sel = " or ".join(residue_selection(r) for r in ligands)
    prot_sel = " or ".join(residue_selection(r) for r in proteins)
    cmd.select(GROUP_SELECTION, f"{lig_sel} or {prot_sel}", enable=0)
    shown_sel = f"{GROUP_SELECTION} and not hydro" if style["hide_hydrogens"] else GROUP_SELECTION
    cmd.show(style["group_representation"], shown_sel)
    cmd.set("stick_radius", style["stick_radius"], OBJECT)
    cmd.color(style["ligand_carbon_color"], f"({lig_sel}) and elem C")
    cmd.color(style["protein_carbon_color"], f"({prot_sel}) and elem C")
    util.cnc(GROUP_SELECTION, _self=cmd)  # cmd may be another instance (recover_view.py)
    cmd.set("cartoon_color", style["cartoon_color_protein"], protein)
    cmd.set("cartoon_color", style["cartoon_color_nucleic"], nucleic)

    # one label per residue, on its first anchor atom
    for label in ligands + proteins:
        sel = residue_selection(label)
        for anchor in style["label_anchor_atoms"]:
            anchor_sel = f"{sel} and name {anchor}"
            if cmd.count_atoms(anchor_sel):
                cmd.label(anchor_sel, json.dumps(label))
                cmd.show("labels", anchor_sel)
                break
    cmd.set("label_size", style["label_size"])
    cmd.set("label_color", style["label_color"], OBJECT)
    cmd.set("label_outline_color", style["label_outline_color"])
    cmd.set("label_font_id", style["label_font_id"])
    cmd.set("label_position", style["label_offset"])
    cmd.set("float_labels", int(style["float_labels"]))
    cmd.set("label_bg_color", style["label_bg_color"])
    cmd.set("label_bg_transparency", style["label_bg_transparency"])
    cmd.set("label_bg_outline", int(style["label_bg_outline"]))

    # dashed lines
    drawn = []
    for k, (lig, prot) in enumerate(pairs):
        if status is not None:
            if status[k] is None:
                continue
            color_key = f"{status[k]}_contact_color"
        elif active[k] or style["show_inactive_contacts"]:
            color_key = None
        else:
            continue
        name = f"{MEASURE_PREFIX}{k:03d}"
        kind = _draw_contact(name, lig, prot, style)
        cmd.color(style[color_key or f"{kind}_contact_color"], name)
        drawn.append((lig, prot, status[k] if status is not None else kind))
    if drawn:
        cmd.hide("labels", f"{MEASURE_PREFIX}*")
        cmd.set("dash_radius", style["dash_radius"])
        cmd.set("dash_gap", style["dash_gap"])
        cmd.set("dash_length", style["dash_length"])

    # user's commands, last
    for line in style["extra_pml_commands"]:
        cmd.do(line)
    return drawn


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def base_view(style):
    """Camera on the group ("orient": principal axes, "reset": default frame)."""
    if style["base_orientation"] == "orient":
        cmd.reset()  # orient depends on the current camera: start from the default one
        cmd.orient(GROUP_SELECTION)
    elif style["base_orientation"] == "reset":
        cmd.reset()
    else:
        raise ValueError(f"Unknown base_orientation {style['base_orientation']!r}")
    cmd.zoom(GROUP_SELECTION, style["zoom_buffer"], complete=1)
    return cmd.get_view()


def rotated_view(base, xang, yang):
    """`base` turned by xang about x, then yang about y (18-float view)."""
    cmd.set_view(base)
    cmd.turn("x", xang)
    cmd.turn("y", yang)
    return cmd.get_view()


LABEL_FREE_REPS = ("lines sticks spheres surface mesh dots cartoon ribbon nonbonded nb_spheres "
                   "ellipsoids cgo dashes angles dihedrals")


def render_png(path, image, style):
    """Ray-trace the current view into `path`.

    Ray tracing ignores float_labels, so with it on the labels are traced on
    a separate transparent layer and pasted on top."""
    width, height, ray = image["width"], image["height"], int(image["ray"])
    if not (style["float_labels"] and ray):
        cmd.png(path, width=width, height=height, ray=ray)
        return

    from PIL import Image

    # never deleted: `scene ..., delete` corrupts memory in PyMOL 3.2
    cmd.scene("_full", "store")
    cmd.hide("labels")
    cmd.png(path, width=width, height=height, ray=1)
    cmd.scene("_full", "recall", animate=0)
    cmd.hide(LABEL_FREE_REPS)
    opaque = cmd.get("ray_opaque_background")
    cmd.set("ray_opaque_background", 0)
    labels_png = path + ".labels.png"
    cmd.png(labels_png, width=width, height=height, ray=1)
    cmd.set("ray_opaque_background", opaque)
    cmd.scene("_full", "recall", animate=0)

    with Image.open(path) as base, Image.open(labels_png) as layer:
        merged = Image.alpha_composite(base.convert("RGBA"), layer.convert("RGBA"))
        merged.save(path)
    os.remove(labels_png)


def tile_key(xang, yang):
    """Name of a menu tile, e.g. 'x030_y150'."""
    return f"x{xang:03d}_y{yang:03d}"


def tile_metadata(menu_meta, key):
    """Metadata of one tile of a menu page, as for a single image."""
    if key not in menu_meta["views"]:
        raise ValueError(f"No tile {key!r} in this menu; tiles: {', '.join(menu_meta['views'])}")
    meta = {k: v for k, v in menu_meta.items() if k != "views"}
    meta.update(xang=int(key[1:4]), yang=int(key[6:9]), view=menu_meta["views"][key])
    return meta


def set_view_command(view):
    return "set_view (" + ", ".join(f"{v:.6f}" for v in view) + ")"


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------

def write_image(path, png_tmp, metadata, image):
    """Save PyMOL's PNG as the final image, with `metadata` embedded."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    text = json.dumps(metadata)
    with Image.open(png_tmp) as img:
        if image["format"] == "png":
            info = PngInfo()
            info.add_text(METADATA_KEY, text, zip=True)
            img.save(path, pnginfo=info, optimize=True)
        elif image["format"] == "jpg":
            img.convert("RGB").save(path, quality=image["jpg_quality"], comment=text.encode())
        else:
            raise ValueError(f"Unknown image format {image['format']!r}")


def read_metadata(path):
    """Metadata embedded by write_image, or None. Pure Python (no PIL)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        pos = 8
        while pos < len(data):
            length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + length]
            if ctype == b"tEXt":
                key, _, value = body.partition(b"\0")
                if key.decode() == METADATA_KEY:
                    return json.loads(value.decode())
            elif ctype == b"zTXt":
                key, _, rest = body.partition(b"\0")
                if key.decode() == METADATA_KEY:
                    return json.loads(zlib.decompress(rest[1:]).decode())
            elif ctype == b"iTXt":
                key, _, rest = body.partition(b"\0")
                if key.decode() == METADATA_KEY:
                    compressed = rest[0]
                    rest = rest[2:]
                    _, _, rest = rest.partition(b"\0")   # language tag
                    _, _, value = rest.partition(b"\0")  # translated keyword
                    if compressed:
                        value = zlib.decompress(value)
                    return json.loads(value.decode())
            pos += 12 + length
        return None
    if data[:2] == b"\xff\xd8":
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF:
            marker = data[pos + 1]
            length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            if marker == 0xFE:  # COM segment
                try:
                    return json.loads(data[pos + 4:pos + 2 + length].decode())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            if marker == 0xDA:  # start of scan
                break
            pos += 2 + length
    return None
