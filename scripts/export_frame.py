"""Write the frame of images made by medoid_snapshots.py as <image>.pdb next to
them, so recover_view.py works without the trajectory (e.g. on a laptop).

    python export_frame.py output/<name>/preview/g00_c11_f00167_x000_y000.png [more ...]
"""
import sys
from pathlib import Path

from pymol import cmd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snapshot_scene as scene  # noqa: E402


def main(images):
    import MDAnalysis as mda

    cmd.feedback("disable", "all", "everything")
    universes = {}
    loaded = None
    for image in images:
        meta = scene.read_metadata(image)
        if meta is None:
            raise SystemExit(f"{image}: no embedded metadata (not made by medoid_snapshots.py?)")
        inputs = meta["inputs"]
        key = (inputs["topology"], inputs["trajectory"])
        if key not in universes:
            universes[key] = mda.Universe(*key)
        if loaded != inputs["topology"]:
            cmd.delete("all")
            scene.load_topology(inputs["topology"])
            loaded = inputs["topology"]
        u = universes[key]
        u.trajectory[meta["frame"]]
        scene.set_coordinates(u.atoms.positions)
        pdb = Path(image).with_suffix(".pdb")
        cmd.save(str(pdb), scene.OBJECT)
        print(f"{pdb} (frame {meta['frame']})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
