# Medoid snapshots

Renders PyMOL images of representative frames of **every interaction group**, from a grid of camera angles. Every image can be restored in PyMOL with its exact view.

1. Read the contact file and keep the informative contacts.
2. Group the contacts by where they sit in space.
3. Per group, pick the frames to show, in one of two modes:
   - **medoid mode** (default): Jaccard distance between frames, HDBSCAN, one medoid per cluster;
   - **reference mode**: the frames whose contacts differ from a reference structure, grouped by which contacts they gained and lost (see [Comparing with a reference](#comparing-with-a-reference)).

## Files (all in `scripts/`)
| file | role |
|---|---|
| `medoid_snapshots.toml` | **all** parameters: inputs, analysis, rotation, image, style |
| `medoid_snapshots.py` | the pipeline |
| `snapshot_scene.py` | builds the PyMOL scene; used by both the pipeline and the recovery tool |
| `recover_view.py` | restores the view of any image |
| `export_frame.py` | writes the frame of an image as a PDB next to it, to restore it on another computer |
| `helper.py` | reading the contact file, grouping, clustering, medoids, reference comparison |
| `labels.py` | residue labels (`ARG152.I`), shared by all scripts |

## Workflow
Use the `py_3.13` env, which has PyMOL, MDAnalysis and sklearn.
```bash
cd /scratch/mgilioli/vis/scripts
python medoid_snapshots.py --config medoid_snapshots.toml --dry-run   # 1. grouping + clustering only (seconds)
python medoid_snapshots.py --config medoid_snapshots.toml --preview   # 2. one image per group -> <outdir>/preview/
#    edit [style], run --preview again, repeat until you like the images
python medoid_snapshots.py --config medoid_snapshots.toml --preview --all-medoids   # 3. one image per medoid, one angle
python medoid_snapshots.py --config medoid_snapshots.toml --menu      # 4. pages of thumbnails of every angle, per medoid
python medoid_snapshots.py --config medoid_snapshots.toml --pick g03_c00_f00000_x045_y135   # 5. full image of a chosen tile
python medoid_snapshots.py --config medoid_snapshots.toml             # (full run: every angle of every medoid as a full image)
```
- **Modes:**

  | mode | medoids | angles | output |
  |---|---|---|---|
  | `--dry-run` | — | — | grouping and clustering only, no images |
  | `--preview` | first medoid of each group | the `[preview]` angle (`xang`, `yang`) | `<outdir>/preview/` |
  | `--preview --all-medoids` | every medoid of every group | the `[preview]` angle only | `<outdir>/preview/` |
  | `--menu` | every medoid of every group | every `[rotation]` angle, as tiles on pages | `<outdir>/menu/` |
  | `--pick TILE...` | the chosen tiles | the tile's angle | `<outdir>/picked/` |
  | no flag (full run) | every medoid of every group | every `[rotation]` angle | `<outdir>/group_GG/images/` |
- **`--preview --all-medoids`:** a quick look at every conformation of every group, all from the same angle. It makes one image per medoid instead of one per medoid per angle, as the full run does. The images go to `<outdir>/preview/` with the usual names (`gGG_cCC_fFFFFF_xXXX_yYYY.png`) and can be restored like any other image.
- **Only the most populated medoids:** `[run] min_prevalence` (or `--min-prevalence 0.05` on the command line) keeps only the medoids whose cluster holds at least that fraction of the trajectory frames; `0` keeps all. It applies to `--preview`, `--preview --all-medoids`, `--menu` and the full run, e.g. `--preview --all-medoids --min-prevalence 0.05`. The `clustering/` files still list every cluster with its `prevalence`, which helps choose the threshold.
- **Menus:** `--menu` makes, for every medoid, pages of `[menu] tiles_per_page` tiles (4 by default, in `columns` columns), one tile per `[rotation]` angle: `<outdir>/menu/gGG_cCC_fFFFFF_pNN.png`. Each tile is labelled with its angle (e.g. `x045_y135`). Pages already on disk are skipped, unless they were drawn with other `[style]`/`[image]` settings.
- **Picking a tile:** `--pick gGG_cCC_fFFFFF_xXXX_yYYY` (one or more) renders the full image of those tiles in `<outdir>/picked/`. It reads everything from the menu pages, so no analysis is rerun. Add `--save-frame` to also save the frame as a PDB next to the image (only if you need to rebuild that conformation without the trajectory, e.g. on another computer).
- `--groups 0 3` processes only those groups, for example one SLURM array task per group. Every job also refreshes `catalog.csv`.
- Rerunning the same command skips images that already exist, so an interrupted run resumes. `--overwrite` re-renders them.
- An output folder holds a single analysis. If you change any setting except `[run]` and `[preview]`, the script stops and asks for a new `output.outdir`, so images made with different settings are never mixed.

## Comparing with a reference
To see which configurations have **contacts different from a reference structure** (e.g. the minimized model), give the contact file of the reference. Every scan in `12-3-analyze_interactions/<scan>/` has one in `<scan>_pdb/contacts.csv`:
```bash
python medoid_snapshots.py --config medoid_snapshots.toml --reference .../head_body_contacts_segidL_pdb/contacts.csv --dry-run
```
or set `[reference] interaction_file`. Every mode (`--preview`, `--menu`, `--pick`, full run) then works on the differences.

1. **Contacts:** all those of the reference, plus the trajectory contacts present in ≥ `min_frequency` (5%) of the frames. Residue names are matched by resid and chain, so `LYS273.F` in the reference is `LYN273.F` in the trajectory.
2. **Groups:** by position, as in medoid mode.
3. **Distance to the reference**, per group and frame: Jaccard = (contacts that differ) / (contacts present in either). Reference {a, b, c} vs frame {a, b, d}: 2/4 = 0.5.
4. **Different frames:** Jaccard > `jaccard_cutoff` (0.3). The others are "like the reference".
5. **Patterns:** different frames with exactly the same contacts, i.e. the same *gained* (frame only) and *lost* (reference only) contacts. Patterns are numbered by size (`c00` = most frequent). The image shows the frame closest to the pattern's average contacts over all groups. Use `--min-prevalence 0.05` to render only patterns with ≥ 5% of the frames.
6. **Images:** dashes colored **grey** = kept, **green** = gained, **red** = lost (`kept/gained/lost_contact_color`). A lost contact is drawn between the closest atoms, to show where it was.

Output, besides the usual files:
- `groups.csv` adds `in_reference` and `frames_differing` (fraction of frames where the contact differs from the reference);
- `group_GG/clustering/patterns.csv`: one row per pattern, with `gained`, `lost`, `n_members`, `prevalence` and the frame shown;
- `group_GG/clustering/labels.csv`: per frame, `jaccard_to_reference` and its pattern (−1 = like the reference);
- `preview/clusters.csv` adds `gained` and `lost`.

A medoid run and a reference run cannot share an output folder.

## Styling (`[style]` in the config)
- **Context:** representations for the whole system (cartoon by default), with their colors and transparency.
- **Group residues:** representation (sticks), with RNA and protein carbons in different colors.
- **Labels:** each displayed residue is labelled in the contact-file format, for example `ARG152.I`. With `float_labels` on, labels are always drawn on top of the structure.
- **Dashed lines** (medoid mode): one per contact that is active in the medoid frame. For reference mode, see above.
  - `polar_contact_color` (green by default): contacts found by PyMOL's polar-contacts tool (the same as *find > polar contacts* in the menu). With hydrogens hidden, the line goes from the donor heavy atom, not its H, so it always touches two drawn atoms.
  - `closest_contact_color` (magenta by default): contacts with no polar partner, for example stacking. The line joins the two closest heavy atoms.
  - `dash_radius`, `dash_gap`, `dash_length`: thickness and pattern of the lines. Keep `dash_gap` below `stick_radius`, so both ends of a line stay inside the sticks.
- `extra_pml`: a `.pml` file with your own PyMOL commands, applied last. Its content is saved with the run, so recovery replays it.
- `extra_commands`: the same, written directly in the config as a multi-line block (`""" ... """`), one PyMOL command per line; run after `extra_pml`.
- `[style.pymol_settings]`: any other PyMOL setting.

## Output
```
<outdir>/
  run_config.json        # resolved configuration + software versions
  groups.csv             # every group: its (ligand, protein) contacts and their frequency
  catalog.csv            # every image: group, cluster, frame, angles, contacts drawn, set_view command, view matrix
  preview/               # --preview (and --preview --all-medoids) images, overwritten each time
    clusters.csv         # one row per preview image: group, cluster, frame, cluster size (n_members, prevalence), active contacts
  menu/                  # --menu pages: gGG_cCC_fFFFFF_pNN.png
  picked/                # --pick images (+ .pdb of the frame with --save-frame)
  group_GG/
    catalog.csv          # this group's rows
    clustering/  labels.csv  medoids.csv  cluster_profiles.csv  summary.json
    images/cCC_fFFFFF/gGG_cCC_fFFFFF_xXXX_yYYY.png
```
A file name such as `g04_c01_f00013_x090_y180.png` means: group 4, cluster 1, medoid frame 13, camera turned 90° about x and then 180° about y. Frames are counted from 0, as in the contact file and MDAnalysis: `f00013` is **state 14** when the trajectory is loaded in PyMOL, which counts states from 1.

To keep disk use low, the run saves no sessions (`.pse`) and no structure files. Each image is rebuilt from the original topology and trajectory, the frame index and the 18-number view matrix. That matrix is embedded in the image and also written in the catalog.

## Restoring a view
```
PyMOL> run /scratch/mgilioli/vis/scripts/recover_view.py
PyMOL> recover /path/to/g04_c01_f00013_x090_y180.png
```
Or from the shell: `pymol /scratch/mgilioli/vis/scripts/recover_view.py -- /path/to/image.png`.

**On another computer** (e.g. your laptop, where the workstation paths stored in the image do not exist):
1. On the workstation, write the frame of the images you want as PDB files next to them (no re-rendering; ~4 MB each):
   `python /scratch/mgilioli/vis/scripts/export_frame.py output/<name>/preview/g00_c11_f00167_x000_y000.png [more ...]`
2. Copy the `.png` and its `.pdb` together, plus `recover_view.py`, `snapshot_scene.py` and `labels.py` (in the same folder).
3. In PyMOL on your computer: `run /path/to/recover_view.py`, then `recover /path/to/g00_c11_f00167_x000_y000.png`. The `.pdb` with the image's name is found automatically.

If you have your own copies of the topology and trajectory, you can skip the PDB: `recover image.png, topology=/local/conf.pdb, trajectory=/local/traj.xtc`. `--pick ... --save-frame` writes the PDB at render time instead of with `export_frame.py`.

For a menu page, give the tile too: `recover page.png, tile=x045_y135`. If a `.pdb` saved with `--save-frame` is next to the image, it is loaded instead of the topology and trajectory.

**Several images in one session:** each recovered image becomes an object named after the image (and its `.pdb`) without the extension, e.g. `g00_c11_f00167_x000_y000`. Its dashed lines go in the group `g00_c11_f00167_x000_y000_contacts` and its selections are renamed `g00_c11_f00167_x000_y000_group_res`, `..._sel1`, ... So you can `recover` one image after another and compare them. Each scene is built in a separate, empty PyMOL instance and only the finished objects are moved into your session, so the `extra_commands` of one image never restyle another, and each `recover` takes about 0.3–0.5 s however many images are loaded. Recovering the same image again replaces it. Images of different inputs can have the same file name (e.g. `g00_c00_f00000_x000_y000.png`): give one of them another name with `recover image.png, name=head_body_g00`.

This rebuilds the same scene (same frame, residues, labels, dashed lines and style) and sets the camera. Everything needed is read from the image itself. If the image has lost its metadata, the information is taken from the output folder's catalog instead. `recover image.png, save=check.png` re-renders the view at the image's own size, so you can compare.

For an even faster check, `catalog.csv` has a `set_view` column: paste it into a PyMOL session that has the same frame loaded.
