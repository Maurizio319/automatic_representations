"""Restore in PyMOL the exact view of an image made by medoid_snapshots.py.

Inside PyMOL:
    run /scratch/mgilioli/vis/scripts/recover_view.py
    recover path/to/g04_c00_f00031_x030_y060.png

From the shell:
    pymol /scratch/mgilioli/vis/scripts/recover_view.py -- path/to/image.png

The scene (frame, residues, labels, dashed contacts, style) and the camera are
read from the metadata embedded in the image. If the metadata is missing (e.g.
the image was converted), they are looked up instead in catalog.csv and
run_config.json of the output folder the image sits in.

`recover image, save=check.png` also re-renders the view to check.png, with
the image's own size and settings, for comparison with the original.

An image made with `--pick ... --save-frame` has its frame saved as a PDB with
the same name next to it: keep the two files together and the view is rebuilt
from them alone, without topology or trajectory (e.g. on another computer).

A menu sheet (medoid_snapshots.py --menu) holds every angle of a medoid:
    recover menu/g03_c00_f00000_p02.png, tile=x030_y150   (the page that shows the tile)
restores the view of that tile (save=... renders it at full size).
"""
import csv
import json
import os
import sys

from pymol import cmd

import inspect  # noqa: E402

# PyMOL's `run` does not set __file__ to this script; the code object knows its file
sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.currentframe().f_code.co_filename)))
import snapshot_scene as scene  # noqa: E402


def _from_catalog(image_path):
    """Rebuild the metadata dict from the output folder's files."""
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
                return {
                    "image": rel, "group": gid, "cluster": int(row["cluster"]), "frame": int(row["frame"]),
                    "xang": int(row["xang"]), "yang": int(row["yang"]),
                    "view": [float(row[f"v{i:02d}"]) for i in range(18)],
                    "pairs": pairs, "active": [c == "1" for c in row["active"]],
                    "inputs": config["inputs"], "image_settings": config["image"], "style": config["style"],
                }
    raise ValueError(f"{rel} is not in the catalogs of {outdir}")


def recover(image, save="", tile=""):
    """recover image [, save=path] [, tile=x030_y150]: rebuild the scene of
    `image` and set its view; for a menu sheet, the view of tile `tile`."""
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

    # a picked image saved with --save-frame has its frame as a PDB next to it
    frame_pdb = os.path.join(os.path.dirname(os.path.abspath(image)), meta.get("frame_pdb", ""))
    if meta.get("frame_pdb") and os.path.exists(frame_pdb):
        scene.load_topology(frame_pdb)
        source += f", structure {meta['frame_pdb']}"
    else:
        scene.load_frame_from_trajectory(meta["inputs"]["topology"], meta["inputs"]["trajectory"], meta["frame"])
    scene.apply_render_settings(meta["style"], meta["image_settings"])
    scene.build_scene([tuple(p) for p in meta["pairs"]], meta["active"], meta["style"])
    cmd.set_view(meta["view"])
    cmd.rebuild()  # place floating labels for this camera (needed when headless)

    print(f"Recovered {meta['image']} ({source}): group {meta['group']}, cluster {meta['cluster']}, "
          f"frame {meta['frame']}, x {meta['xang']} deg, y {meta['yang']} deg")
    print(scene.set_view_command(meta["view"]))

    if save:
        settings = meta["image_settings"]
        scene.render_png(save, settings, meta["style"])
        print(f"Re-rendered to {save}")


cmd.extend("recover", recover)

# `pymol recover_view.py -- image.png [save.png]`
if __name__ in ("__main__", "pymol") and len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    recover(*sys.argv[1:4])
