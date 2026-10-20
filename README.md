# Medoid snapshots

Renders PyMOL images of the medoid frames of **every interaction group**, from a grid of camera angles. Every image can be restored in PyMOL with its exact view.

It uses the method from `notebooks/prova_01contacts.ipynb`:
1. Read the contact file and drop uninformative contacts.
2. Group the contacts by where they sit in space.
3. For each group: compute the Jaccard distance between frames, cluster with HDBSCAN, and take one medoid per cluster.

## Files (all in `scripts/`)
| file | role |
|---|---|
| `medoid_snapshots.toml` | **all** parameters: inputs, analysis, rotation, image, style |
| `medoid_snapshots.py` | the pipeline |
| `snapshot_scene.py` | builds the PyMOL scene; used by both the pipeline and the recovery tool |
| `recover_view.py` | restores the view of any image |
| `helper.py` | reading the contact file, grouping, clustering, medoids |

## Workflow
Use the `py_3.13` env, which has PyMOL, MDAnalysis and sklearn.
```bash
cd /scratch/mgilioli/vis/scripts
python medoid_snapshots.py --config medoid_snapshots.toml --dry-run   # 1. grouping + clustering only (seconds)
python medoid_snapshots.py --config medoid_snapshots.toml --preview   # 2. one image per group -> <outdir>/preview/
#    edit [style], run --preview again, repeat until you like the images
python medoid_snapshots.py --config medoid_snapshots.toml             # 3. full run
```
- `--groups 0 3` processes only those groups, for example one SLURM array task per group. Every job also refreshes `catalog.csv`.
- Rerunning the same command skips images that already exist, so an interrupted run resumes. `--overwrite` re-renders them.
- An output folder holds a single analysis. If you change any setting except `[run]` and `[preview]`, the script stops and asks for a new `output.outdir`, so images made with different settings are never mixed.

## Styling (`[style]` in the config)
- **Context:** representations for the whole system (cartoon by default), with their colors and transparency.
- **Group residues:** representation (sticks), with RNA and protein carbons in different colors.
- **Labels:** each displayed residue is labelled in the contact-file format, for example `ARG152.I`. With `float_labels` on, labels are always drawn on top of the structure.
- **Dashed lines:** one per contact that is active in the medoid frame.
  - `polar_contact_color` (red): contacts found by PyMOL's polar-contacts tool (the same as *find > polar contacts* in the menu).
  - `closest_contact_color` (magenta): contacts with no polar partner, for example stacking or hydrophobic contacts. The line joins the two closest heavy atoms.
- `extra_pml`: a `.pml` file with your own PyMOL commands, applied last. Its content is saved with the run, so recovery replays it.
- `extra_commands`: the same, written directly in the config as a multi-line block (`""" ... """`), one PyMOL command per line; run after `extra_pml`.
- `[style.pymol_settings]`: any other PyMOL setting.

## Output
```
<outdir>/
  run_config.json        # resolved configuration + software versions
  groups.csv             # every group: its (ligand, protein) contacts and their frequency
  catalog.csv            # every image: group, cluster, frame, angles, contacts drawn, set_view command, view matrix
  preview/               # --preview images (overwritten each time)
  group_GG/
    catalog.csv          # this group's rows
    clustering/  labels.csv  medoids.csv  cluster_profiles.csv  summary.json
    images/cCC_fFFFFF/gGG_cCC_fFFFFF_xXXX_yYYY.png
```
A file name such as `g04_c01_f00013_x090_y180.png` means: group 4, cluster 1, medoid frame 13, camera turned 90° about x and then 180° about y.

To keep disk use low, the run saves no sessions (`.pse`) and no structure files. Each image is rebuilt from the original topology and trajectory, the frame index and the 18-number view matrix. That matrix is embedded in the image and also written in the catalog.

## Restoring a view
```
PyMOL> run /scratch/mgilioli/vis/scripts/recover_view.py
PyMOL> recover /path/to/g04_c01_f00013_x090_y180.png
```
Or from the shell: `pymol /scratch/mgilioli/vis/scripts/recover_view.py -- /path/to/image.png`.

This rebuilds the same scene (same frame, residues, labels, dashed lines and style) and sets the camera. Everything needed is read from the image itself. If the image has lost its metadata, the information is taken from the output folder's catalog instead. `recover image.png, save=check.png` re-renders the view at the image's own size, so you can compare.

For an even faster check, `catalog.csv` has a `set_view` column: paste it into a PyMOL session that has the same frame loaded.
