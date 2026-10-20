import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

def read_ipf_alt(path):
    """Read a wide ProLIF interaction-fingerprint CSV, as written by
    scripts/run_evaluate_int_RNA_other_strand.py: '# sel1: ...' / '# sel2:
    ...' comment lines, then a 3-level (ligand, protein, interaction) column
    header and one 0/1 row per frame.

    The interaction type is dropped: a residue pair is 1 in a frame if any
    interaction (of any type, any atom pair) is present.

    Returns a DataFrame with one row per frame and one (ligand, protein)
    column per residue pair, values 0/1.
    """
    with open(path) as f:
        n_comments = 0
        for line in f:
            if not line.startswith("#"):
                break
            n_comments += 1

    df = pd.read_csv(path, skiprows=n_comments, header=[0, 1, 2], index_col=0)
    # OR together every column sharing the same (ligand, protein) pair
    return df.T.groupby(level=[0, 1]).max().T

def read_ipf(path):
    """Read a wide ProLIF interaction-fingerprint CSV, as written by
    scripts/run_evaluate_int_RNA_other_strand.py: '# sel1: ...' / '# sel2:
    ...' comment lines, then a 3-level (ligand, protein, interaction) column
    header and one 0/1 row per frame.

    Returns {(ligand, protein): {interaction_type: Series of 0/1 per frame}}.
    """
    with open(path) as f:
        n_comments = 0
        for line in f:
            if not line.startswith("#"):
                break
            n_comments += 1

    df = pd.read_csv(path, skiprows=n_comments, header=[0, 1, 2], index_col=0)
    out = {}
    for pair in sorted(set(df.columns.droplevel(2))):
        pair_cols = [c for c in df.columns if c[:2] == pair]
        types = {}
        for t in sorted(set(c[2] for c in pair_cols)):
            cols = [c for c in pair_cols if c[2] == t]
            # a residue pair can have >1 atom-pair combo for the same
            # interaction type -- OR them together
            types[t] = df[cols].max(axis=1)
        out[pair] = types
    return out

def frobenius_dist(v1, v2, cap=None):
        """Frobenius (Euclidean) distance between two same-shaped arrays.

        `v1`/`v2` are (n_residues, 3) base-COM arrays. Without a cap this is
        the ordinary Frobenius norm of their difference. With a cap, each
        residue's own displacement is log-tapered past `cap` *before* being
        folded into the norm -- so one residue that moved a lot can't by
        itself dominate the frame-pair distance, but it still contributes
        more than a residue that barely moved (unlike a hard clip, which
        would flatten every large displacement to the same value).
        """
        a = np.asarray(v1)
        b = np.asarray(v2)
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        diff = a - b
        if cap is None:
            return np.linalg.norm(diff)

        per_atom = np.linalg.norm(diff, axis=-1)  # (n_residues,)
        over = per_atom > cap
        per_atom[over] = cap + np.log1p(per_atom[over] - cap)
        return np.linalg.norm(per_atom)

def run_clustering(dist, minsize, outfile=None, verbose=True):
    # copy=True: sklearn defaults to copy=False, which overwrites `dist` in place
    # with mutual-reachability distances and corrupts later uses of it
    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=minsize, copy=True)
    labels = clusterer.fit_predict(dist)
    if verbose:
        print("HDBSCAN labels:", labels)
    if outfile:
        np.save(outfile, labels)
    return labels

    
def evaluate_clustering(dist, labels):
    """Silhouette score for a precomputed-distance clustering.

    Computed directly on `dist` (metric="precomputed"), not on an MDS
    embedding, so it reflects the actual pairwise distances rather than a
    lossy 2D/low-dim projection of them. Range [-1, 1]: near 1 means
    points sit closer to their own cluster than to the next-nearest one,
    near 0 means clusters overlap, negative means points are probably
    mislabeled.

    Points labeled -1 (HDBSCAN noise) are dropped first -- noise isn't a
    real cluster to score against. Returns NaN if fewer than 2 clusters
    remain after dropping noise (silhouette is undefined in that case).
    """
    from sklearn.metrics import silhouette_score

    dist = np.asarray(dist)
    labels = np.asarray(labels)
    mask = labels != -1

    if len(np.unique(labels[mask])) < 2:
        print("Silhouette score: undefined (fewer than 2 non-noise clusters)")
        return float("nan")

    # silhouette_score requires an exact-zero diagonal; guard against
    # floating-point noise there regardless of how `dist` was built.
    sub_dist = dist[np.ix_(mask, mask)].copy()
    np.fill_diagonal(sub_dist, 0)

    score = silhouette_score(sub_dist, labels[mask], metric="precomputed")
    print(f"Silhouette score: {score:.3f}")
    return score


# ---------------------------------------------------------------------------
# Interaction grouping and medoids (moved from notebooks/prova_01contacts.ipynb)
# ---------------------------------------------------------------------------

LABEL_RE = re.compile(r"([A-Za-z0-9]+?)(-?\d+)\.(\w+)")


def parse_label(label):
    """'ARG152.I' -> ('ARG', 152, 'I'); 'C4.C' -> ('C', 4, 'C')."""
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Cannot parse residue label {label!r}")
    resname, resid, chain = match.groups()
    return resname, int(resid), chain


def filter_columns(df, threshold):
    """Keep only the residue-pair columns whose contact frequency lies in
    [threshold, 1 - threshold]: always-on or always-off contacts carry no
    information for clustering."""
    frequency = df.apply(
        lambda col: col.map(
            lambda x: bool(x) if isinstance(x, list)
            else bool(x) if pd.notna(x)
            else False
        ).mean()
    )
    keep = frequency.between(threshold, 1 - threshold)
    return df.loc[:, keep]


def residue_label(residue):
    """Residue -> label in the interaction-file format, e.g. 'ARG152.I'."""
    return f"{residue.resname}{residue.resid}.{residue.atoms.chainIDs[0]}"


def get_residue(u, label, label_to_residue=None):
    """Label like 'C10.C' or 'ARG152.I' -> MDAnalysis Residue."""
    if label_to_residue is not None and label in label_to_residue:
        return label_to_residue[label]
    resname, resid, chain = parse_label(label)
    sel = u.select_atoms(f"chainID {chain} and resname {resname} and resid {resid}")
    if len(sel.residues) != 1:
        raise ValueError(f"Label {label} matches {len(sel.residues)} residues")
    return sel.residues[0]


def build_interaction_groups(interactions, u, group_distance, starting_frame):
    """Group (ligand, protein) residue pairs by where they sit in space.

    Each interaction is placed at the midpoint of its two residues' centers
    of geometry -- in the first frame if `starting_frame`, else averaged over
    the whole trajectory -- and the points are cut by average-linkage
    hierarchical clustering at `group_distance` (Angstrom).

    Returns a list of groups, each a list of (ligand, protein) label pairs.
    """
    label_to_residue = {residue_label(res): res for res in u.residues}
    atom_pairs = [
        (get_residue(u, a, label_to_residue).atoms, get_residue(u, b, label_to_residue).atoms)
        for a, b in interactions
    ]

    def midpoints():
        return np.array([(a.center_of_geometry() + b.center_of_geometry()) / 2 for a, b in atom_pairs])

    if starting_frame:
        u.trajectory[0]
        positions = midpoints()
    else:
        positions = np.zeros((len(interactions), 3))
        for _ in u.trajectory:
            positions += midpoints()
        positions /= len(u.trajectory)

    if len(interactions) == 1:
        group_ids = np.array([1])
    else:
        group_ids = fcluster(linkage(pdist(positions), method="average"), group_distance, criterion="distance")
    return [
        [inter for inter, gid in zip(interactions, group_ids) if gid == k] for k in sorted(set(group_ids))
    ]


def jaccard_distance(contacts):
    """(n_frames, n_contacts) 0/1 array -> (n_frames, n_frames) Jaccard
    distance: fraction of differing contacts among those present in either
    frame, so contacts absent in both frames don't count as similarity."""
    return squareform(pdist(np.asarray(contacts, dtype=bool), metric="jaccard"))


def compute_medoids(dist, labels):
    """One medoid per non-noise cluster: the member with the smallest total
    within-cluster distance. On ties, argmin returns the first member, i.e.
    the earliest frame.

    Returns a list of dicts: cluster, medoid_row (row in `dist`), n_members.
    """
    rows = []
    for cluster_id in sorted(set(labels) - {-1}):  # skip HDBSCAN noise
        members = np.flatnonzero(labels == cluster_id)
        within = dist[np.ix_(members, members)]
        rows.append({
            "cluster": int(cluster_id),
            "medoid_row": int(members[np.argmin(within.sum(axis=1))]),
            "n_members": len(members),
        })
    return rows
