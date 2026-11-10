"""PyMOL snapshots of representative frames of every interaction group.

  1. read the contact file and keep the informative contacts
  2. group the contacts by where they sit in space
  3. per group, pick representative frames:
     - medoid mode: Jaccard distance between frames -> HDBSCAN -> medoids
     - reference mode ([reference] interaction_file or --reference): frames
       that differ from a reference structure, grouped by contacts gained/lost
  4. render them from a grid of camera angles, recording every view

  python medoid_snapshots.py --config medoid_snapshots.toml --dry-run   # steps 1-3
  python medoid_snapshots.py --config medoid_snapshots.toml --preview [--all-medoids]
  python medoid_snapshots.py --config medoid_snapshots.toml --menu
  python medoid_snapshots.py --config medoid_snapshots.toml --pick g03_c00_f00000_x030_y150
  python medoid_snapshots.py --config medoid_snapshots.toml   # full run

Restore any image's view with recover_view.py.
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

from helper import (build_interaction_groups, compute_medoids, contact_status, filter_columns,
                    jaccard_distance, read_ipf_alt, read_reference, reference_columns,
                    reference_patterns, run_clustering)

ANALYSIS_SECTIONS = ("inputs", "output", "filter", "grouping", "clustering", "reference")
RENDER_SECTIONS = ("rotation", "image", "style")
REFERENCE_DEFAULTS = {"interaction_file": "", "jaccard_cutoff": 0.3, "min_frequency": 0.05}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path, reference=None):
    """Read the TOML config: resolve relative paths, inline style.extra_pml.
    reference: overrides reference.interaction_file."""
    path = Path(path).resolve()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    def resolve(p):
        return str((path.parent / p).resolve()) if p else p

    for key in ("topology", "trajectory", "interaction_file"):
        cfg["inputs"][key] = resolve(cfg["inputs"][key])
    cfg["output"]["outdir"] = resolve(cfg["output"]["outdir"])
    cfg["reference"] = {**REFERENCE_DEFAULTS, **cfg.get("reference", {})}
    if reference is not None:
        cfg["reference"]["interaction_file"] = str(Path(reference).resolve())
    else:
        cfg["reference"]["interaction_file"] = resolve(cfg["reference"]["interaction_file"])

    style = cfg["style"]
    style["extra_pml"] = resolve(style["extra_pml"])
    style["extra_pml_commands"] = []
    if style["extra_pml"]:
        with open(style["extra_pml"]) as f:
            style["extra_pml_commands"] = [
                line.strip() for line in f if line.strip() and not line.strip().startswith("#")
            ]
    style["extra_pml_commands"] += [
        line.strip() for line in style.get("extra_commands", "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    rot = cfg["rotation"]
    for key in ("step", "x_start", "x_stop", "y_start", "y_stop"):
        if not isinstance(rot[key], int):
            raise ValueError(f"rotation.{key} must be an integer (it is used in file names)")
    files = [(f"inputs.{key}", cfg["inputs"][key]) for key in ("topology", "trajectory", "interaction_file")]
    if is_reference_mode(cfg):
        files.append(("reference.interaction_file", cfg["reference"]["interaction_file"]))
    for name, file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{name}: {file} does not exist")
    cfg["config_file"] = str(path)
    return cfg


def is_reference_mode(cfg):
    return bool(cfg["reference"]["interaction_file"])


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
    """Save the resolved config as <outdir>/run_config.json. One folder holds one
    analysis: sections already saved must match; missing ones are added."""
    path = outdir / "run_config.json"
    record = {s: cfg[s] for s in sections}
    if not is_reference_mode(cfg):
        record["reference"] = {"interaction_file": ""}  # unused settings never block a run
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        if "filter" in old["config"]:  # folders made before [reference] are medoid runs
            old["config"].setdefault("reference", {"interaction_file": ""})
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
    """Write via temporary file + rename (safe with parallel jobs)."""
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
# Analysis
# ---------------------------------------------------------------------------

def pair_names(pairs, mask):
    return ";".join(f"{lig}-{prot}" for (lig, prot), on in zip(pairs, mask) if on)


def run_analysis(cfg, u, selected, outdir=None):
    """Steps 1-3. Writes the analysis files when `outdir` is given.

    Returns (df, groups): the contact table and, per selected group,
    {"pairs", "medoids": [{cluster, medoid_row, frame, n_members, prevalence}]}
    (+ "reference": per pair, in the reference). Only medoids with prevalence
    >= run.min_prevalence are returned; the files list all of them.
    """
    min_prevalence = cfg["run"].get("min_prevalence", 0.0)
    reference_mode = is_reference_mode(cfg)
    path = cfg["inputs"]["interaction_file"]
    with open(path) as f:
        header = next(line for line in f if not line.startswith("#"))
    if "," not in header:  # only the frame column
        raise SystemExit(f"{path} has no contacts: nothing to group or render")
    if reference_mode:
        ref_pairs = read_reference(cfg["reference"]["interaction_file"], u)
        df = reference_columns(read_ipf_alt(path), ref_pairs, cfg["reference"]["min_frequency"])
    else:
        df = filter_columns(read_ipf_alt(path), cfg["filter"]["threshold"])
    if df.shape[1] == 0:
        raise SystemExit(f"{path}: no contact left after filtering: nothing to group or render")
    if len(df) != len(u.trajectory) or not (df.index == np.arange(len(df))).all():
        raise SystemExit(
            f"The interaction file has {len(df)} rows (index {df.index[0]}..{df.index[-1]}) but the "
            f"trajectory has {len(u.trajectory)} frames: rows must be trajectory frames 0..N-1."
        )
    print(f"{len(df)} frames, {df.shape[1]} contacts"
          + (f" ({len(ref_pairs)} in the reference)" if reference_mode else ""), flush=True)

    all_groups = build_interaction_groups(
        list(df.columns), u, cfg["grouping"]["group_distance"], cfg["grouping"]["starting_frame"]
    )
    if selected is None:
        selected = list(range(len(all_groups)))
    unknown = sorted(set(selected) - set(range(len(all_groups))))
    if unknown:
        raise SystemExit(f"Groups {unknown} do not exist: there are {len(all_groups)} groups (0..{len(all_groups) - 1})")

    if outdir is not None:
        rows = []
        for gid, pairs in enumerate(all_groups):
            for pair in pairs:
                row = {"group": gid, "ligand": pair[0], "protein": pair[1],
                       "frequency": df[pair].astype(bool).mean()}
                if reference_mode:
                    row["in_reference"] = pair in ref_pairs
                    row["frames_differing"] = float((df[pair].astype(bool) != (pair in ref_pairs)).mean())
                rows.append(row)
        atomic_write_csv(outdir / "groups.csv", pd.DataFrame(rows))

    all_contacts = df.to_numpy(dtype=bool)
    groups = {}
    for gid in selected:
        pairs = all_groups[gid]
        contacts = df[pairs].to_numpy(dtype=bool)
        if reference_mode:
            ref = np.array([p in ref_pairs for p in pairs])
            dist, labels, medoids = reference_patterns(
                contacts, ref, cfg["reference"]["jaccard_cutoff"], all_contacts)
        else:
            dist = jaccard_distance(contacts)
            labels = run_clustering(dist, cfg["clustering"]["min_cluster_size"], verbose=False)
            medoids = compute_medoids(dist, labels)
        for m in medoids:
            m["frame"] = int(df.index[m["medoid_row"]])
            m["prevalence"] = m["n_members"] / len(df)
        kept = [m for m in medoids if m["prevalence"] >= min_prevalence]
        groups[gid] = {"pairs": pairs, "medoids": kept}
        unassigned = float(np.mean(labels == -1))
        if reference_mode:
            groups[gid]["reference"] = ref.tolist()
        print(f"group {gid:2d}: {len(pairs):2d} contacts, {len(medoids)} "
              + (f"patterns, like the reference {unassigned:.0%}" if reference_mode
                 else f"clusters, noise {unassigned:.0%}")
              + f", frames {[m['frame'] for m in kept]}"
              + (f" ({len(kept)} with prevalence >= {min_prevalence})" if min_prevalence > 0 else ""),
              flush=True)

        if outdir is not None:
            cdir = group_dir(outdir, gid) / "clustering"
            cdir.mkdir(parents=True, exist_ok=True)
            summary = {"group": gid, "n_frames": len(df), "n_clusters": len(medoids), "pairs": pairs}
            columns = ["cluster", "frame", "medoid_row", "n_members", "prevalence"]
            if reference_mode:
                atomic_write_csv(cdir / "labels.csv", pd.DataFrame(
                    {"frame": df.index, "jaccard_to_reference": dist.round(4), "cluster": labels}))
                table = pd.DataFrame(medoids, columns=columns + ["gained", "lost"])
                table["gained"] = [pair_names(pairs, m["gained"]) for m in medoids]
                table["lost"] = [pair_names(pairs, m["lost"]) for m in medoids]
                atomic_write_csv(cdir / "patterns.csv", table)
                summary.update(n_frames_like_reference=int(np.sum(labels == -1)),
                               reference=pair_names(pairs, ref))
            else:
                atomic_write_csv(cdir / "labels.csv", pd.DataFrame({"frame": df.index, "cluster": labels}))
                atomic_write_csv(cdir / "medoids.csv", pd.DataFrame(medoids)[columns])
                profiles = pd.DataFrame(
                    [contacts[labels == m["cluster"]].mean(axis=0) for m in medoids],
                    columns=[f"{lig}|{prot}" for lig, prot in pairs],
                )
                profiles.insert(0, "cluster", [m["cluster"] for m in medoids])
                atomic_write_csv(cdir / "cluster_profiles.csv", profiles)
                summary["noise_fraction"] = unassigned
            atomic_write_text(cdir / "summary.json", json.dumps(summary, indent=2))
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
    cmd.rebuild()  # headless: floating labels follow the new camera only after a rebuild
    raw = os.path.join(tmpdir, "raw.png")
    scene.render_png(raw, cfg["image"], cfg["style"])
    scene.write_image(path, raw, metadata, cfg["image"])


def image_metadata(cfg, gid, medoid, pairs, active, status, rel_path, camera=None):
    """Everything needed to rebuild the image. camera: (xang, yang, view);
    None for a menu page, whose views are added per tile."""
    meta = {
        "image": rel_path, "group": gid, "cluster": medoid["cluster"], "frame": medoid["frame"],
        "pairs": [list(p) for p in pairs], "active": [bool(a) for a in active],
        "inputs": cfg["inputs"], "image_settings": cfg["image"], "style": cfg["style"],
    }
    if status is not None:
        meta["status"] = status
    if camera is not None:
        xang, yang, view = camera
        meta.update(xang=xang, yang=yang, view=list(view))
    return meta


def scene_tasks(df, groups):
    """(frame, gid, medoid) of every medoid, sorted by frame (each frame loaded once)."""
    tasks = [(m["frame"], gid, m) for gid, g in groups.items() for m in g["medoids"]]
    return sorted(tasks, key=lambda t: (t[0], t[1]))


def start_pymol(cfg):
    from pymol import cmd
    import snapshot_scene as scene

    cmd.feedback("disable", "all", "everything")
    scene.load_topology(cfg["inputs"]["topology"])
    scene.apply_render_settings(cfg["style"], cfg["image"])
    return scene


def prepare_scene(scene, u, df, groups, gid, medoid, cfg, current_frame=None):
    """Load the medoid's frame (unless already loaded) and build its scene.
    Returns (pairs, active, status, drawn, base view)."""
    if medoid["frame"] != current_frame:
        u.trajectory[medoid["frame"]]
        scene.set_coordinates(u.atoms.positions)
    g = groups[gid]
    pairs = g["pairs"]
    active = df.iloc[medoid["medoid_row"]][pairs].astype(bool).tolist()
    status = contact_status(active, g["reference"]) if "reference" in g else None
    drawn = scene.build_scene(pairs, active, cfg["style"], status)
    return pairs, active, status, drawn, scene.base_view(cfg["style"])


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
            pairs, active, status, drawn, base = prepare_scene(scene, u, df, groups, gid, medoid, cfg,
                                                               current_frame)
            current_frame = frame
            mdir = medoid_dir(outdir, gid, medoid["cluster"], frame, cfg)
            mdir.mkdir(parents=True, exist_ok=True)
            contacts_drawn = ";".join(f"{lig}-{prot}:{kind}" for lig, prot, kind in drawn)

            for xang, yang in grid:
                view = scene.rotated_view(base, xang, yang)
                path = mdir / image_name(gid, medoid["cluster"], frame, xang, yang, cfg)
                rel = str(path.relative_to(outdir))
                if cfg["run"]["overwrite"] or not path.exists():
                    meta = image_metadata(cfg, gid, medoid, pairs, active, status, rel, (xang, yang, view))
                    render_one(path, view, meta, cfg, tmpdir)
                    rendered += 1
                row = {
                    "image": rel, "group": gid, "cluster": medoid["cluster"], "frame": frame,
                    "medoid_row": medoid["medoid_row"], "n_members": medoid["n_members"],
                    "xang": xang, "yang": yang, "contacts_drawn": contacts_drawn,
                    "active": "".join("1" if a else "0" for a in active),
                }
                if status is not None:
                    row["status"] = ";".join(s or "-" for s in status)
                rows[gid].append({**row, "set_view": scene.set_view_command(view),
                                  **{f"v{i:02d}": v for i, v in enumerate(view)}})
                done += 1
                if done % cfg["run"]["progress_every"] == 0 or done == n_images:
                    print(f"  {done}/{n_images} images ({rendered} rendered, {done - rendered} already on disk)",
                          flush=True)

            remaining[gid] -= 1
            if remaining[gid] == 0:
                atomic_write_csv(group_dir(outdir, gid) / "catalog.csv", pd.DataFrame(rows.pop(gid)))


def merge_catalogs(outdir):
    """<outdir>/catalog.csv = every group's catalog.csv."""
    parts = sorted(outdir.glob("group_*/catalog.csv"))
    if parts:
        atomic_write_csv(outdir / "catalog.csv", pd.concat([pd.read_csv(p) for p in parts], ignore_index=True))
        print(f"Catalog: {outdir / 'catalog.csv'}", flush=True)


def render_preview(cfg, u, df, groups, outdir, all_medoids=False):
    """First medoid of each group (every medoid with all_medoids) at the
    preview angle, in <outdir>/<preview.outdir_name>/, listed in clusters.csv."""
    scene = start_pymol(cfg)
    pdir = outdir / cfg["preview"]["outdir_name"]
    pdir.mkdir(parents=True, exist_ok=True)
    xang, yang = cfg["preview"]["xang"], cfg["preview"]["yang"]
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gid, g in groups.items():
            if not g["medoids"]:
                print(f"group {gid:2d}: no clusters (above run.min_prevalence), nothing to preview", flush=True)
                continue
            for medoid in g["medoids"] if all_medoids else g["medoids"][:1]:
                pairs, active, status, _, base = prepare_scene(scene, u, df, groups, gid, medoid, cfg)
                view = scene.rotated_view(base, xang, yang)
                path = pdir / image_name(gid, medoid["cluster"], medoid["frame"], xang, yang, cfg)
                meta = image_metadata(cfg, gid, medoid, pairs, active, status,
                                      str(path.relative_to(outdir)), (xang, yang, view))
                render_one(path, view, meta, cfg, tmpdir)
                row = {
                    "image": path.name, "group": gid, "cluster": medoid["cluster"], "frame": medoid["frame"],
                    "n_members": medoid["n_members"], "n_frames": len(df),
                    "prevalence": round(medoid["prevalence"], 4),
                    "active_contacts": pair_names(pairs, active),
                }
                if status is not None:
                    row["gained"] = pair_names(pairs, [s == "gained" for s in status])
                    row["lost"] = pair_names(pairs, [s == "lost" for s in status])
                rows.append(row)
                print(f"preview: {path}  (cluster of {medoid['n_members']} frames, "
                      f"{medoid['prevalence']:.1%})", flush=True)

    # keep the rows of earlier previews still on disk (e.g. other --groups)
    table = pdir / "clusters.csv"
    new = pd.DataFrame(rows)
    if table.exists():
        old = pd.read_csv(table)
        old = old[~old["image"].isin(new["image"] if rows else []) & old["image"].map(lambda n: (pdir / n).exists())]
        new = pd.concat([old, new], ignore_index=True)
    if len(new):
        new = new.sort_values(["group", "cluster"])
        atomic_write_csv(table, new)
        print(f"Cluster sizes: {table}", flush=True)


def menu_name(gid, cluster, frame, page, cfg):
    return f"g{gid:02d}_c{cluster:02d}_{frame_tag(frame, cfg)}_p{page:02d}.png"


def is_current_page(path, cfg):
    """Menu page exists and was drawn with the current inputs, image and style."""
    if not path.exists():
        return False
    import snapshot_scene as scene
    meta = scene.read_metadata(path)
    return meta is not None and all(
        meta.get(key) == cfg[section]
        for key, section in (("inputs", "inputs"), ("image_settings", "image"), ("style", "style")))


def render_menu(cfg, u, df, groups, outdir):
    """Pages of thumbnails of every [rotation] angle, per medoid
    (<outdir>/<menu.outdir_name>/gGG_cCC_fFFFFF_pNN.png). Each page embeds the
    scene and every tile's view, for --pick and recover_view.py."""
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
            todo = [p for p, path in enumerate(paths) if cfg["run"]["overwrite"] or not is_current_page(path, cfg)]
            if not todo:
                print(f"menu {k}/{len(tasks)}: g{gid:02d} c{medoid['cluster']:02d} already on disk", flush=True)
                continue
            pairs, active, status, _, base = prepare_scene(scene, u, df, groups, gid, medoid, cfg, current_frame)
            current_frame = frame
            for p in todo:
                page, path = pages[p], paths[p]
                rows = -(-len(page) // cols)
                sheet = Image.new("RGB", (cols * tw, rows * th), "white")
                views = {}
                for i, (xang, yang) in enumerate(page):
                    view = scene.rotated_view(base, xang, yang)
                    cmd.rebuild()
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

                meta = image_metadata(cfg, gid, medoid, pairs, active, status, str(path.relative_to(outdir)))
                meta.update(page=p + 1, n_pages=len(pages), views=views)
                sheet.save(sheet_png)
                scene.write_image(path, sheet_png, meta, {"format": "png"})
            print(f"menu {k}/{len(tasks)}: {paths[0].name[:-8]}_p01..p{len(pages):02d}", flush=True)


def find_menu_tile(mdir, sheet_stem, key, read_metadata):
    """Metadata of the menu page of `sheet_stem` that holds tile `key`."""
    pages = sorted(mdir.glob(f"{sheet_stem}_p*.png"))
    if not pages:
        raise SystemExit(f"{sheet_stem}: no menu pages in {mdir} (run --menu first)")
    for page in pages:
        meta = read_metadata(page)
        if meta is not None and key in meta["views"]:
            return meta
    raise SystemExit(f"{sheet_stem}: no tile {key} in its menu pages")


def render_pick(cfg, u, names, outdir, save_frame=False):
    """Full image of each picked menu tile, in <outdir>/picked/, rebuilt from the
    menu page. save_frame: also write the frame as <image>.pdb."""
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
            scene.build_scene([tuple(p) for p in meta["pairs"]], meta["active"], meta["style"], meta.get("status"))
            path = pdir / f"{sheet_stem}_{x}_{y}.{meta['image_settings']['format']}"
            meta["image"] = str(path.relative_to(outdir))
            if save_frame:
                pdb = path.with_suffix(".pdb")
                cmd.save(str(pdb), scene.OBJECT)
                meta["frame_pdb"] = pdb.name
            render_one(path, meta["view"], meta, {"image": meta["image_settings"], "style": meta["style"]}, tmpdir)
            print(f"picked: {path}" + (f" (+ {meta['frame_pdb']})" if save_frame else ""), flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="TOML configuration file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="analysis only, no rendering")
    mode.add_argument("--preview", action="store_true", help="one image per group, to tune [style]")
    mode.add_argument("--menu", action="store_true", help="pages of thumbnails of every angle, per medoid")
    mode.add_argument("--pick", nargs="+", metavar="TILE",
                      help="full image of menu tiles, e.g. g03_c00_f00000_x030_y150")
    parser.add_argument("--all-medoids", action="store_true", help="with --preview: every medoid, not only the first")
    parser.add_argument("--save-frame", action="store_true", help="with --pick: also save the frame as a PDB")
    parser.add_argument("--reference", metavar="CSV",
                        help="contact file of a reference structure: override reference.interaction_file")
    parser.add_argument("--groups", type=int, nargs="+", help="override run.groups")
    parser.add_argument("--min-prevalence", type=float, metavar="FRACTION", help="override run.min_prevalence")
    parser.add_argument("--overwrite", action="store_true", help="override run.overwrite")
    args = parser.parse_args()
    if args.all_medoids and not args.preview:
        parser.error("--all-medoids goes with --preview")

    cfg = load_config(args.config, args.reference)
    if args.groups is not None:
        cfg["run"]["groups"] = args.groups
    if args.overwrite:
        cfg["run"]["overwrite"] = True
    if args.min_prevalence is not None:
        cfg["run"]["min_prevalence"] = args.min_prevalence
    selected = cfg["run"]["groups"] or None

    import MDAnalysis as mda

    outdir = Path(cfg["output"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(cfg["inputs"]["topology"], cfg["inputs"]["trajectory"])

    if args.pick:  # everything is in the menu pages
        render_pick(cfg, u, args.pick, outdir, args.save_frame)
        return
    if args.menu:
        df, groups = run_analysis(cfg, u, selected)
        render_menu(cfg, u, df, groups, outdir)
        return
    if args.preview:  # writes only the previews, so [style] can change freely
        df, groups = run_analysis(cfg, u, selected)
        render_preview(cfg, u, df, groups, outdir, args.all_medoids)
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
