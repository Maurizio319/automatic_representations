"""Restore in PyMOL the exact scene and view of an image made by medoid_snapshots.py.

    PyMOL> run /scratch/mgilioli/vis/scripts/recover_view.py
    PyMOL> recover path/to/g04_c00_f00031_x030_y060.png [, save=check.png] [, name=...]
    PyMOL> recover menu/g03_c00_f00000_p02.png, tile=x030_y150
    shell> pymol recover_view.py -- path/to/image.png

The scene comes from the image's metadata (or, if lost, from the output
folder's catalog). The structure comes from <image>.pdb next to it if present
(export_frame.py, --save-frame), else from topology=/trajectory=, else from the
paths stored in the image. Each image becomes its own object, named after the
file, so several can be loaded side by side. See README.md.
"""
import csv
import json
import os
import sys

import pymol2
from pymol import cmd

import inspect  # noqa: E402

# PyMOL's `run` does not set __file__
sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.currentframe().f_code.co_filename)))
import snapshot_scene as scene  # noqa: E402


def _from_catalog(image_path):
    """Metadata rebuilt from the output folder's catalog and run config."""
    image_path = os.path.abspath(image_path)
    outdir = os.path.dirname(image_path)
    while not os.path.exists(os.path.join(outdir, "run_config.json")):
        parent = os.path.dirname(outdir)
        if parent == outdir:
            raise ValueError(f"{image_path}: no embedded metadata and no run_config.json in any parent folder")
        outdir = parent
    rel = os.path.relpath(image_path, outdir)
    with open(os.path.join(outdir, "run_config.json")) as f:
        config = json.load(f)["config"]

    for catalog in (os.path.join(outdir, rel.split(os.sep)[0], "catalog.csv"), os.path.join(outdir, "catalog.csv")):
        if not os.path.exists(catalog):
            continue
        with open(catalog) as f:
            for row in csv.DictReader(f):
                if row["image"] != rel:
                    continue
                gid = int(row["group"])
                summary = os.path.join(outdir, f"group_{gid:02d}", "clustering", "summary.json")
                with open(summary) as g:
                    pairs = json.load(g)["pairs"]
                status = row.get("status")
                meta = {
                    "image": rel, "group": gid, "cluster": int(row["cluster"]), "frame": int(row["frame"]),
                    "xang": int(row["xang"]), "yang": int(row["yang"]),
                    "view": [float(row[f"v{i:02d}"]) for i in range(18)],
                    "pairs": pairs, "active": [c == "1" for c in row["active"]],
                    "inputs": config["inputs"], "image_settings": config["image"], "style": config["style"],
                }
                if status:
                    meta["status"] = [None if s == "-" else s for s in status.split(";")]
                return meta
    raise ValueError(f"{rel} is not in the catalogs of {outdir}")


def _selection_atoms(_self):
    """{selection: {object: [indices]}}: a partial session does not carry selections."""
    atoms = {}
    for sel in _self.get_names("selections"):
        per_object = {}
        for obj, index in _self.index(sel):
            per_object.setdefault(obj, []).append(index)
        atoms[sel] = per_object
    return atoms


def _restore_selections(atoms):
    for sel, per_object in atoms.items():
        parts = []
        for obj, indices in per_object.items():
            ranges, start = [], None  # consecutive indices as a-b ranges
            for i, index in enumerate(sorted(indices)):
                if start is None:
                    start = prev = index
                elif index != prev + 1:
                    ranges.append(f"{start}-{prev}")
                    start = index
                prev = index
                if i == len(indices) - 1:
                    ranges.append(f"{start}-{prev}")
            parts.append(f"({obj} and index {'+'.join(ranges)})")
        cmd.select(sel, " or ".join(parts) if parts else "none", enable=0)


def _global_settings(_self):
    return {setting: _self.get(setting) for setting in _self.setting.get_name_list()}


def recover(image, save="", tile="", topology="", trajectory="", name=""):
    """recover image [, save=path] [, tile=x030_y150] [, topology=..., trajectory=...] [, name=...]"""
    meta = scene.read_metadata(image)
    source = "embedded metadata"
    if meta is None:
        meta = _from_catalog(image)
        source = "catalog"
    if "views" in meta:
        if not tile:
            raise ValueError(f"{image} is a menu sheet: give tile=..., one of {', '.join(meta['views'])}")
        meta = scene.tile_metadata(meta, tile)
        source += f", tile {tile}"

    frame_pdb = os.path.splitext(os.path.abspath(image))[0] + ".pdb"
    if not name:
        name = os.path.splitext(os.path.basename(image))[0]
        if tile:  # gGG_cCC_fFFFFF_pNN -> gGG_cCC_fFFFFF_<tile>
            name = f"{name.rsplit('_', 1)[0]}_{tile}"

    cmd.delete(f"{name}_*")  # replace an earlier recovery of this image
    cmd.delete(name)
    if not os.path.exists(frame_pdb):
        topology = topology or meta["inputs"]["topology"]
        trajectory = trajectory or meta["inputs"]["trajectory"]
        for path in (topology, trajectory):
            if not os.path.exists(path):
                raise ValueError(
                    f"{path} does not exist on this computer. Either put {os.path.basename(frame_pdb)} next to "
                    f"the image (python export_frame.py <image> on the workstation), or give your local "
                    f"copies: recover {image}, topology=..., trajectory=...")

    # build in a separate PyMOL instance, so extra_commands touch this image only
    instance = pymol2.PyMOL()
    instance.start()
    main_cmd, scene.cmd = scene.cmd, instance.cmd
    try:
        pcmd = instance.cmd
        pcmd.feedback("disable", "all", "everything")
        defaults = _global_settings(pcmd)
        if os.path.exists(frame_pdb):
            scene.load_topology(frame_pdb)
            source += f", structure {os.path.basename(frame_pdb)}"
        else:
            scene.load_frame_from_trajectory(topology, trajectory, meta["frame"])
        scene.apply_render_settings(meta["style"], meta["image_settings"])
        scene.build_scene([tuple(p) for p in meta["pairs"]], meta["active"], meta["style"], meta.get("status"))
        pcmd.set_view(meta["view"])

        if save:
            pcmd.rebuild()  # headless: place floating labels for this camera
            scene.render_png(save, meta["image_settings"], meta["style"])
            print(f"Re-rendered to {save}")

        # name everything after the image
        for sel in pcmd.get_names("selections"):
            pcmd.set_name(sel, f"{name}_{sel}")
        contacts = [obj for obj in pcmd.get_names("objects") if obj.startswith(scene.MEASURE_PREFIX)]
        for obj in contacts:
            pcmd.set_name(obj, f"{name}_{obj}")
        pcmd.set_name(scene.OBJECT, name)
        if contacts:
            pcmd.group(f"{name}_contacts", " ".join(f"{name}_{obj}" for obj in contacts))

        session = pcmd.get_session()
        selections = _selection_atoms(pcmd)
        changed = {k: v for k, v in _global_settings(pcmd).items() if v != defaults.get(k)}
    finally:
        scene.cmd = main_cmd
        instance.stop()

    # move the objects into this session, with the global settings the scene changed
    cmd.set_session(session, partial=1)
    for setting, value in changed.items():
        if cmd.get(setting) == value:
            continue  # setting it redraws every object
        try:
            cmd.set(setting, value)
        except Exception:  # read-only or session-level settings
            pass
    _restore_selections(selections)
    cmd.set_view(meta["view"])
    cmd.rebuild(f"{name} or {name}_*")

    print(f"Recovered {meta['image']} ({source}): group {meta['group']}, cluster {meta['cluster']}, "
          f"frame {meta['frame']} (0-based; state {meta['frame'] + 1} of the trajectory loaded in PyMOL), x {meta['xang']} deg, y {meta['yang']} deg")
    print(scene.set_view_command(meta["view"]))
    print(f"Object: {name}")


cmd.extend("recover", recover)

# `pymol recover_view.py -- image.png [save.png]`
if __name__ in ("__main__", "pymol") and len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    recover(*sys.argv[1:4])
