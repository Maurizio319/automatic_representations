"""Rotating PyMOL snapshots of the medoid frames of every interaction group.

Pipeline (method from notebooks/prova_01contacts.ipynb):
  1. read the RNA-protein contact fingerprint and drop uninformative contacts
  2. group the contacts by where they sit in space
  3. for every group: Jaccard distance between frames -> HDBSCAN -> medoids
  4. for every medoid of every group: render the group's interactions from a
     grid of camera angles, cataloguing the exact view of every image

All parameters come from a TOML config (medoid_snapshots.toml). Typical use:

  python medoid_snapshots.py --config medoid_snapshots.toml --dry-run   # steps 1-3 only
  python medoid_snapshots.py --config medoid_snapshots.toml --preview   # one image per group, to tune [style]
  python medoid_snapshots.py --config medoid_snapshots.toml --menu      # pages of thumbnails of all angles, per medoid
  python medoid_snapshots.py --config medoid_snapshots.toml --pick g03_c00_f00000_x030_y150   # full image of a tile
  python medoid_snapshots.py --config medoid_snapshots.toml             # full run

Restore the view of any image afterwards with recover_view.py.
"""
import argparse
import datetime
import json
import os
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from helper import (build_interaction_groups, compute_medoids, filter_columns,
                    jaccard_distance, read_ipf_alt, run_clustering)

ANALYSIS_SECTIONS = ("inputs", "output", "filter", "grouping", "clustering")
RENDER_SECTIONS = ("rotation", "image", "style")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
    """Read the TOML config, resolve relative paths against its directory and
    inline the content of style.extra_pml (so the run config alone is enough
    to rebuild any view)."""
    path = Path(path).resolve()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    def resolve(p):
        return str((path.parent / p).resolve()) if p else p

    for key in ("topology", "trajectory", "interaction_file"):
        cfg["inputs"][key] = resolve(cfg["inputs"][key])
    cfg["output"]["outdir"] = resolve(cfg["output"]["outdir"])

    style = cfg["style"]
    style["extra_pml"] = resolve(style["extra_pml"])
    style["extra_pml_commands"] = []
    if style["extra_pml"]:
        with open(style["extra_pml"]) as f:
            style["extra_pml_commands"] = [
                line.strip() for line in f if line.strip() and not line.strip().startswith("#")
            ]
    # inline commands of the config, run after those of the .pml file
    # (optional, so templates written before this option still work)
    style["extra_pml_commands"] += [
        line.strip() for line in style.get("extra_commands", "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    rot = cfg["rotation"]
    for key in ("step", "x_start", "x_stop", "y_start", "y_stop"):
        if not isinstance(rot[key], int):
            raise ValueError(f"rotation.{key} must be an integer (it is used in file names)")
    for key in ("topology", "trajectory", "interaction_file"):
        if not os.path.exists(cfg["inputs"][key]):
            raise FileNotFoundError(f"inputs.{key}: {cfg['inputs'][key]} does not exist")
    cfg["config_file"] = str(path)
    return cfg


def software_versions():
    import MDAnalysis
    import sklearn
    versions = {"numpy": np.__version__, "pandas": pd.__version__,
                "MDAnalysis": MDAnalysis.__version__, "sklearn": sklearn.__version__}
    try:
        from pymol import cmd
        versions["pymol"] = cmd.get_version()[0]
    except ImportError:
        pass
    return versions


def write_run_config(cfg, outdir, sections):
    """Save the resolved config as <outdir>/run_config.json.

    An output folder holds exactly one analysis: if run_config.json already
    exists, every section it shares with this run must be identical, so
    images rendered with different settings never get mixed. Sections it
    lacks (e.g. rendering after a --dry-run) are added.
    """
    path = outdir / "run_config.json"
    record = {s: cfg[s] for s in sections}
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        for s in sections:
            if s in old["config"] and old["config"][s] != record[s]:
                raise SystemExit(
                    f"[{s}] differs from the run already in {outdir} (see {path}).\n"
                    f"Use a new output.outdir, or restore the old [{s}] settings."
                )
        old["config"].update(record)
        record = old["config"]
        created = old["created"]
    else:
        created = datetime.datetime.now().isoformat(timespec="seconds")
    atomic_write_text(path, json.dumps({
        "created": created,
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_file": cfg["config_file"],
        "software": software_versions(),
        "config": record,
    }, indent=2))


def atomic_write_text(path, text):
    """Write via a temporary file + rename, so parallel jobs never leave a
    half-written file."""
    tmp = Path(f"{path}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_csv(path, df):
    atomic_write_text(path, df.to_csv(index=False))


# ---------------------------------------------------------------------------
# Paths and names
# ---------------------------------------------------------------------------

def group_dir(outdir, gid):
    return outdir / f"group_{gid:02d}"


def frame_tag(frame, cfg):
    return f"f{frame:0{cfg['output']['frame_digits']}d}"


def medoid_dir(outdir, gid, cluster, frame, cfg):
    return group_dir(outdir, gid) / "images" / f"c{cluster:02d}_{frame_tag(frame, cfg)}"


def image_name(gid, cluster, frame, xang, yang, cfg):
    return (f"g{gid:02d}_c{cluster:02d}_{frame_tag(frame, cfg)}"
            f"_x{xang:03d}_y{yang:03d}.{cfg['image']['format']}")


# ---------------------------------------------------------------------------
# Analysis: grouping, clustering, medoids
# ---------------------------------------------------------------------------

def run_analysis(cfg, u, selected, outdir=None):
    """Steps 1-3 for all groups. Writes groups.csv and each selected group's
    clustering/ folder when `outdir` is given.

    Returns (df, groups): `df` the filtered contact table, `groups` a dict
    gid -> {"pairs": [...], "medoids": [ {cluster, medoid_row, frame,
    n_members, prevalence}, ... ]} for the selected groups.
    """
    df = filter_columns(read_ipf_alt(cfg["inputs"]["interaction_file"]), cfg["filter"]["threshold"])
    if len(df) != len(u.trajectory) or not (df.index == np.arange(len(df))).all():
        raise SystemExit(
            f"The interaction file has {len(df)} rows (index {df.index[0]}..{df.index[-1]}) but the "
            f"trajectory has {len(u.trajectory)} frames: rows must be trajectory frames 0..N-1."
        )
    print(f"{len(df)} frames, {df.shape[1]} informative contacts", flush=True)

    all_groups = build_interaction_groups(
        list(df.columns), u, cfg["grouping"]["group_distance"], cfg["grouping"]["starting_frame"]
    )
    if selected is None:
        selected = list(range(len(all_groups)))
    unknown = sorted(set(selected) - set(range(len(all_groups))))
    if unknown:
        raise SystemExit(f"Groups {unknown} do not exist: there are {len(all_groups)} groups (0..{len(all_groups) - 1})")

    if outdir is not None:
        atomic_write_csv(outdir / "groups.csv", pd.DataFrame(
            [{"group": gid, "ligand": lig, "protein": prot, "frequency": df[(lig, prot)].astype(bool).mean()}
             for gid, pairs in enumerate(all_groups) for lig, prot in pairs]
        ))

    groups = {}
    for gid in selected:
        pairs = all_groups[gid]
        contacts = df[pairs].to_numpy(dtype=bool)
        dist = jaccard_distance(contacts)
        labels = run_clustering(dist, cfg["clustering"]["min_cluster_size"], verbose=False)
        medoids = compute_medoids(dist, labels)
        for m in medoids:
            m["frame"] = int(df.index[m["medoid_row"]])
            m["prevalence"] = m["n_members"] / len(df)
        groups[gid] = {"pairs": pairs, "medoids": medoids}
        noise = float(np.mean(labels == -1))
        print(f"group {gid:2d}: {len(pairs):2d} contacts, {len(medoids)} clusters, "
              f"noise {noise:.0%}, medoid frames {[m['frame'] for m in medoids]}", flush=True)

        if outdir is not None:
            cdir = group_dir(outdir, gid) / "clustering"
            cdir.mkdir(parents=True, exist_ok=True)
            atomic_write_csv(cdir / "labels.csv", pd.DataFrame({"frame": df.index, "cluster": labels}))
            atomic_write_csv(cdir / "medoids.csv", pd.DataFrame(medoids)[
                ["cluster", "frame", "medoid_row", "n_members", "prevalence"]])
            profiles = pd.DataFrame(
                [contacts[labels == m["cluster"]].mean(axis=0) for m in medoids],
                columns=[f"{lig}|{prot}" for lig, prot in pairs],
            )
            profiles.insert(0, "cluster", [m["cluster"] for m in medoids])
            atomic_write_csv(cdir / "cluster_profiles.csv", profiles)
            atomic_write_text(cdir / "summary.json", json.dumps({
                "group": gid, "n_frames": len(df), "n_clusters": len(medoids),
                "noise_fraction": noise, "pairs": pairs,
            }, indent=2))
    return df, groups


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def angle_grid(cfg):
    rot = cfg["rotation"]
    return [(x, y)
            for x in range(rot["x_start"], rot["x_stop"], rot["step"])
            for y in range(rot["y_start"], rot["y_stop"], rot["step"])]


def render_one(path, view, metadata, cfg, tmpdir):
    """Ray-trace the current scene at `view` into `path`, with metadata."""
    from pymol import cmd
    import snapshot_scene as scene

    cmd.set_view(view)
    # headless PyMOL never redraws, so floating labels would keep the screen
    # positions of the previous view: rebuild so they follow this camera
    cmd.rebuild()
    raw = os.path.join(tmpdir, "raw.png")
    scene.render_png(raw, cfg["image"], cfg["style"])
    scene.write_image(path, raw, metadata, cfg["image"])


def image_metadata(cfg, gid, medoid, pairs, active, xang, yang, view, rel_path):
    """Everything needed to rebuild this exact image, embedded in the file."""
    return {
        "image": rel_path, "group": gid, "cluster": medoid["cluster"], "frame": medoid["frame"],
        "xang": xang, "yang": yang, "view": list(view),
        "pairs": [list(p) for p in pairs], "active": [bool(a) for a in active],
        "inputs": cfg["inputs"], "image_settings": cfg["image"], "style": cfg["style"],
    }


def scene_tasks(df, groups):
    """(frame, gid, medoid) for every medoid of every group, sorted by frame
    so each frame's coordinates are set once even when it is the medoid of
    several groups."""
    tasks = [(m["frame"], gid, m) for gid, g in groups.items() for m in g["medoids"]]
    return sorted(tasks, key=lambda t: (t[0], t[1]))


def start_pymol(cfg):
    from pymol import cmd
    import snapshot_scene as scene

    cmd.feedback("disable", "all", "everything")
    scene.load_topology(cfg["inputs"]["topology"])
    scene.apply_render_settings(cfg["style"], cfg["image"])
    return scene


def render_all(cfg, u, df, groups, outdir):
    scene = start_pymol(cfg)
    grid = angle_grid(cfg)
    tasks = scene_tasks(df, groups)
    n_images = len(tasks) * len(grid)
    print(f"Rendering {len(tasks)} medoids x {len(grid)} views = {n_images} images", flush=True)

    rows = defaultdict(list)
    remaining = defaultdict(int)
    for _, gid, _ in tasks:
        remaining[gid] += 1

    done = rendered = 0
    current_frame = None
    with tempfile.TemporaryDirectory() as tmpdir:
        for frame, gid, medoid in tasks:
            if frame != current_frame:
                u.trajectory[frame]
                scene.set_coordinates(u.atoms.positions)
                current_frame = frame

            pairs = groups[gid]["pairs"]
            active = df.iloc[medoid["medoid_row"]][pairs].astype(bool).tolist()
            drawn = scene.build_scene(pairs, active, cfg["style"])
            base = scene.base_view(cfg["style"])
            mdir = medoid_dir(outdir, gid, medoid["cluster"], frame, cfg)
            mdir.mkdir(parents=True, exist_ok=True)
            contacts_drawn = ";".join(f"{lig}-{prot}:{kind}" for lig, prot, kind in drawn)

            for xang, yang in grid:
                view = scene.rotated_view(base, xang, yang)
                path = mdir / image_name(gid, medoid["cluster"], frame, xang, yang, cfg)
                rel = str(path.relative_to(outdir))
                if cfg["run"]["overwrite"] or not path.exists():
                    meta = image_metadata(cfg, gid, medoid, pairs, active, xang, yang, view, rel)
                    render_one(path, view, meta, cfg, tmpdir)
                    rendered += 1
                rows[gid].append({
                    "image": rel, "group": gid, "cluster": medoid["cluster"], "frame": frame,
                    "medoid_row": medoid["medoid_row"], "n_members": medoid["n_members"],
                    "xang": xang, "yang": yang, "contacts_drawn": contacts_drawn,
                    "active": "".join("1" if a else "0" for a in active),
                    "set_view": scene.set_view_command(view),
                    **{f"v{i:02d}": v for i, v in enumerate(view)},
                })
                done += 1
                if done % cfg["run"]["progress_every"] == 0 or done == n_images:
                    print(f"  {done}/{n_images} images ({rendered} rendered, {done - rendered} already on disk)",
                          flush=True)

            remaining[gid] -= 1
            if remaining[gid] == 0:
                atomic_write_csv(group_dir(outdir, gid) / "catalog.csv", pd.DataFrame(rows.pop(gid)))


def merge_catalogs(outdir):
    """<outdir>/catalog.csv = every group's catalog.csv, in group order."""
    parts = sorted(outdir.glob("group_*/catalog.csv"))
    if parts:
        atomic_write_csv(outdir / "catalog.csv", pd.concat([pd.read_csv(p) for p in parts], ignore_index=True))
        print(f"Catalog: {outdir / 'catalog.csv'}", flush=True)


def render_preview(cfg, u, df, groups, outdir):
    """One image per group (its first medoid, at the preview angles), in
    <outdir>/<preview.outdir_name>/, overwritten every time: a quick loop to
    tune the [style] section before the full run."""
    scene = start_pymol(cfg)
    pdir = outdir / cfg["preview"]["outdir_name"]
    pdir.mkdir(parents=True, exist_ok=True)
    xang, yang = cfg["preview"]["xang"], cfg["preview"]["yang"]
    with tempfile.TemporaryDirectory() as tmpdir:
        for gid, g in groups.items():
            if not g["medoids"]:
                print(f"group {gid:2d}: no clusters, nothing to preview", flush=True)
                continue
            medoid = g["medoids"][0]
            u.trajectory[medoid["frame"]]
            scene.set_coordinates(u.atoms.positions)
            active = df.iloc[medoid["medoid_row"]][g["pairs"]].astype(bool).tolist()
            scene.build_scene(g["pairs"], active, cfg["style"])
            view = scene.rotated_view(scene.base_view(cfg["style"]), xang, yang)
            path = pdir / image_name(gid, medoid["cluster"], medoid["frame"], xang, yang, cfg)
            meta = image_metadata(cfg, gid, medoid, g["pairs"], active, xang, yang, view,
                                  str(path.relative_to(outdir)))
            render_one(path, view, meta, cfg, tmpdir)
            print(f"preview: {path}", flush=True)


def menu_name(gid, cluster, frame, page, cfg):
    return f"g{gid:02d}_c{cluster:02d}_{frame_tag(frame, cfg)}_p{page:02d}.png"


def render_menu(cfg, u, df, groups, outdir):
    """Menu pages for every medoid of every group, in <outdir>/<menu.outdir_name>/:
    a thumbnail of every angle of the [rotation] grid, menu.tiles_per_page
    tiles per page in menu.columns columns, each tile named by its angle
    (e.g. x030_y150); pages g03_c00_f00000_p01.png, _p02.png, ...

    Each page embeds the medoid's scene and the camera of each of its tiles,
    so a tile can be turned into its full image (--pick) or opened in PyMOL
    (recover_view.py, tile=...) with exactly the view shown."""
    from pymol import cmd
    from PIL import Image, ImageDraw, ImageFont

    scene = start_pymol(cfg)
    grid = angle_grid(cfg)
    menu = cfg["menu"]
    per_page, cols = menu["tiles_per_page"], menu["columns"]
    pages = [grid[i:i + per_page] for i in range(0, len(grid), per_page)]
    tw, th = menu["tile_width"], menu["tile_height"]
    tile_image = {**cfg["image"], "width": tw, "height": th}
    font = ImageFont.load_default(size=menu["tile_label_size"])
    mdir = outdir / menu["outdir_name"]
    mdir.mkdir(parents=True, exist_ok=True)
    tasks = scene_tasks(df, groups)
    print(f"Menu: {len(tasks)} medoids x {len(grid)} tiles = {len(tasks)} x {len(pages)} pages", flush=True)

    current_frame = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tile_png, sheet_png = os.path.join(tmpdir, "tile.png"), os.path.join(tmpdir, "sheet.png")
        for k, (frame, gid, medoid) in enumerate(tasks, 1):
            paths = [mdir / menu_name(gid, medoid["cluster"], frame, p, cfg) for p in range(1, len(pages) + 1)]
            todo = [p for p, path in enumerate(paths) if cfg["run"]["overwrite"] or not path.exists()]
            if not todo:
                print(f"menu {k}/{len(tasks)}: g{gid:02d} c{medoid['cluster']:02d} already on disk", flush=True)
                continue
            if frame != current_frame:
                u.trajectory[frame]
                scene.set_coordinates(u.atoms.positions)
                current_frame = frame

            pairs = groups[gid]["pairs"]
            active = df.iloc[medoid["medoid_row"]][pairs].astype(bool).tolist()
            scene.build_scene(pairs, active, cfg["style"])
            base = scene.base_view(cfg["style"])
            for p in todo:
                page, path = pages[p], paths[p]
                rows = -(-len(page) // cols)
                sheet = Image.new("RGB", (cols * tw, rows * th), "white")
                views = {}
                for i, (xang, yang) in enumerate(page):
                    view = scene.rotated_view(base, xang, yang)
                    cmd.rebuild()  # floating labels follow this camera (headless)
                    scene.render_png(tile_png, tile_image, cfg["style"])
                    key = scene.tile_key(xang, yang)
                    with Image.open(tile_png) as tile:
                        tile = tile.convert("RGB")
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle(draw.textbbox((6, 6), key, font=font), fill="white")
                    draw.text((6, 6), key, fill="black", font=font)
                    draw.rectangle((0, 0, tw - 1, th - 1), outline="grey")
                    sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
                    views[key] = list(view)

                meta = image_metadata(cfg, gid, medoid, pairs, active, None, None, (),
                                      str(path.relative_to(outdir)))
                for key in ("xang", "yang", "view"):
                    del meta[key]
                meta.update(page=p + 1, n_pages=len(pages), views=views)
                sheet.save(sheet_png)
                scene.write_image(path, sheet_png, meta, {"format": "png"})
            print(f"menu {k}/{len(tasks)}: {paths[0].name[:-8]}_p01..p{len(pages):02d}", flush=True)


def find_menu_tile(mdir, sheet_stem, key, read_metadata):
    """Metadata of the menu page of medoid `sheet_stem` that holds tile `key`."""
    pages = sorted(mdir.glob(f"{sheet_stem}_p*.png"))
    if not pages:
        raise SystemExit(f"{sheet_stem}: no menu pages in {mdir} (run --menu first)")
    for page in pages:
        meta = read_metadata(page)
        if meta is not None and key in meta["views"]:
            return meta
    raise SystemExit(f"{sheet_stem}: no tile {key} in its menu pages")


def render_pick(cfg, u, names, outdir, save_frame=False):
    """Full image of each picked tile, e.g. g03_c00_f00000_x030_y150, in
    <outdir>/picked/. Scene, style and camera come from the menu sheet, so the
    image is exactly the tile that was chosen; it carries the same metadata
    as any rendered image (recover_view.py works on it).

    save_frame: also write the frame as <image>.pdb next to the image and
    record it in the metadata, so recover_view.py rebuilds the view from the
    image and that PDB alone, without the trajectory."""
    from pymol import cmd
    import snapshot_scene as scene

    cmd.feedback("disable", "all", "everything")
    scene.load_topology(cfg["inputs"]["topology"])
    pdir = outdir / "picked"
    pdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in names:
            sheet_stem, x, y = Path(name).stem.rsplit("_", 2)
            menu_meta = find_menu_tile(outdir / cfg["menu"]["outdir_name"], sheet_stem, f"{x}_{y}",
                                       scene.read_metadata)
            meta = scene.tile_metadata(menu_meta, f"{x}_{y}")
            for key in ("page", "n_pages"):
                meta.pop(key, None)
            u.trajectory[meta["frame"]]
            scene.set_coordinates(u.atoms.positions)
            scene.apply_render_settings(meta["style"], meta["image_settings"])
            scene.build_scene([tuple(p) for p in meta["pairs"]], meta["active"], meta["style"])
            path = pdir / f"{sheet_stem}_{x}_{y}.{meta['image_settings']['format']}"
            meta["image"] = str(path.relative_to(outdir))
            if save_frame:
                pdb = path.with_suffix(".pdb")
                cmd.save(str(pdb), scene.OBJECT)
                meta["frame_pdb"] = pdb.name  # next to the image
            render_one(path, meta["view"], meta, {"image": meta["image_settings"], "style": meta["style"]}, tmpdir)
            print(f"picked: {path}" + (f" (+ {meta['frame_pdb']})" if save_frame else ""), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="TOML configuration file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="grouping and clustering only, no rendering")
    mode.add_argument("--preview", action="store_true", help="one image per group, to tune the [style] section")
    mode.add_argument("--menu", action="store_true",
                      help="one sheet per medoid with a thumbnail of every angle, to choose views from")
    mode.add_argument("--pick", nargs="+", metavar="TILE",
                      help="render the full image of menu tiles, e.g. g03_c00_f00000_x030_y150")
    parser.add_argument("--save-frame", action="store_true",
                        help="with --pick: also save the frame as a PDB next to the image, so "
                             "recover_view.py can rebuild it without the trajectory")
    parser.add_argument("--groups", type=int, nargs="+", help="override run.groups (e.g. one SLURM task per group)")
    parser.add_argument("--overwrite", action="store_true", help="override run.overwrite: re-render existing images")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.groups is not None:
        cfg["run"]["groups"] = args.groups
    if args.overwrite:
        cfg["run"]["overwrite"] = True
    selected = cfg["run"]["groups"] or None

    import MDAnalysis as mda

    outdir = Path(cfg["output"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(cfg["inputs"]["topology"], cfg["inputs"]["trajectory"])

    if args.pick:
        # everything needed is in the menu sheets: no analysis
        render_pick(cfg, u, args.pick, outdir, args.save_frame)
        return
    if args.menu:
        df, groups = run_analysis(cfg, u, selected)
        render_menu(cfg, u, df, groups, outdir)
        return
    if args.preview:
        # previews write nothing but the preview images, so styling can be
        # iterated freely without touching the run in outdir
        df, groups = run_analysis(cfg, u, selected)
        render_preview(cfg, u, df, groups, outdir)
        return

    sections = ANALYSIS_SECTIONS if args.dry_run else ANALYSIS_SECTIONS + RENDER_SECTIONS
    write_run_config(cfg, outdir, sections)
    df, groups = run_analysis(cfg, u, selected, outdir)
    if args.dry_run:
        return
    render_all(cfg, u, df, groups, outdir)
    merge_catalogs(outdir)


if __name__ == "__main__":
    main()
