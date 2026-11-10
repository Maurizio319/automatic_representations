from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import HDBSCAN

from labels import parse_label


# ---------------------------------------------------------------------------
# Reading the contact file
# ---------------------------------------------------------------------------

def _read_fingerprint(path):
    """ProLIF fingerprint CSV: '#' comment lines, a 3-level (ligand, protein,
    interaction) header, one 0/1 row per frame."""
    with open(path) as f:
        n_comments = 0
        for line in f:
            if not line.startswith("#"):
                break
            n_comments += 1
    return pd.read_csv(path, skiprows=n_comments, header=[0, 1, 2], index_col=0)


def read_ipf_alt(path):
    """Contact file -> one 0/1 column per (ligand, protein) pair, any interaction type."""
    df = _read_fingerprint(path)
    # OR together every column sharing the same (ligand, protein) pair
    return df.T.groupby(level=[0, 1]).max().T


def filter_columns(df, threshold):
    """Keep the pairs present in [threshold, 1 - threshold] of the frames."""
    frequency = df.astype(bool).mean()
    return df.loc[:, frequency.between(threshold, 1 - threshold)]


# ---------------------------------------------------------------------------
# Grouping and medoids
# ---------------------------------------------------------------------------

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
    """Group (ligand, protein) pairs by the position of their midpoint (frame 0
    if `starting_frame`, else the trajectory average): average-linkage
    clustering cut at `group_distance` Angstrom. Returns a list of pair lists."""
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
    """(n_frames, n_contacts) 0/1 -> (n_frames, n_frames) Jaccard distance."""
    return squareform(pdist(np.asarray(contacts, dtype=bool), metric="jaccard"))


def run_clustering(dist, minsize, outfile=None, verbose=True):
    # copy=True: otherwise sklearn overwrites `dist` in place
    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=minsize, copy=True)
    labels = clusterer.fit_predict(dist)
    if verbose:
        print("HDBSCAN labels:", labels)
    if outfile:
        np.save(outfile, labels)
    return labels


def compute_medoids(dist, labels):
    """One medoid per non-noise cluster (smallest total distance; ties: earliest).
    Returns [{cluster, medoid_row, n_members}]."""
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


# ---------------------------------------------------------------------------
# Comparison with a reference structure
# ---------------------------------------------------------------------------

def read_reference(path, u):
    """(ligand, protein) pairs of a one-frame contact file, renamed to the
    topology's labels by resid and chain (e.g. LYS273.F -> LYN273.F)."""
    ref = read_ipf_alt(path)
    if len(ref) != 1:
        raise ValueError(f"{path}: a reference has one frame, not {len(ref)}")
    by_position = defaultdict(list)
    for res in u.residues:
        by_position[(res.resid, res.atoms.chainIDs[0])].append(residue_label(res))

    def rename(label):
        _, resid, chain = parse_label(label)
        found = by_position.get((resid, chain), [])
        if len(found) != 1:
            raise ValueError(f"{path}: {label} matches {len(found)} residues of the topology")
        return found[0]

    return {(rename(a), rename(b)) for (a, b), on in ref.iloc[0].items() if on}


def reference_columns(df, ref_pairs, min_frequency):
    """The reference pairs plus the pairs present in >= min_frequency of the
    frames; reference pairs never seen in the trajectory are added as 0."""
    frequency = df.astype(bool).mean()
    out = df[[c for c in df.columns if c in ref_pairs or frequency[c] >= min_frequency]].copy()
    for pair in sorted(ref_pairs - set(df.columns)):
        out[pair] = 0
    return out


def reference_patterns(contacts, ref, cutoff, all_contacts):
    """Group the frames whose Jaccard distance from `ref` exceeds `cutoff` by
    their exact contacts, i.e. by what they gained and lost.

    contacts: (n_frames, n_pairs) of the group; all_contacts: of every pair,
    used to pick the frame closest to the pattern's average.
    Returns (dist, labels, patterns): labels -1 = like the reference;
    patterns largest first, [{cluster, medoid_row, n_members, gained, lost}].
    """
    contacts = np.asarray(contacts, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    dist = cdist(contacts, ref[None], "jaccard")[:, 0] if ref.size else np.zeros(len(contacts))
    rows_of = defaultdict(list)
    for row in np.flatnonzero(dist > cutoff):
        rows_of[contacts[row].tobytes()].append(row)

    labels = np.full(len(contacts), -1)
    patterns = []
    for cid, rows in enumerate(sorted(rows_of.values(), key=lambda r: (-len(r), r[0]))):
        labels[rows] = cid
        members = np.asarray(all_contacts[rows], dtype=float)
        typical = rows[int(np.argmin(np.abs(members - members.mean(axis=0)).sum(axis=1)))]
        frame = contacts[rows[0]]
        patterns.append({"cluster": cid, "medoid_row": int(typical), "n_members": len(rows),
                         "gained": frame & ~ref, "lost": ref & ~frame})
    return dist, labels, patterns


def contact_status(active, ref):
    """Per pair: "kept", "gained", "lost" or None (in neither)."""
    return ["kept" if a and r else "gained" if a else "lost" if r else None
            for a, r in zip(active, ref)]


# ---------------------------------------------------------------------------
# Notebook utilities: used by notebooks/, not by the pipeline
# ---------------------------------------------------------------------------

def read_ipf(path):
    """Contact file -> {(ligand, protein): {interaction_type: 0/1 Series}}."""
    df = _read_fingerprint(path)
    out = {}
    for pair in sorted(set(df.columns.droplevel(2))):
        pair_cols = [c for c in df.columns if c[:2] == pair]
        types = {}
        for t in sorted(set(c[2] for c in pair_cols)):
            cols = [c for c in pair_cols if c[2] == t]
            # OR the atom-pair columns of the same type
            types[t] = df[cols].max(axis=1)
        out[pair] = types
    return out


def frobenius_dist(v1, v2, cap=None):
    """Frobenius distance of two (n_residues, 3) arrays; with `cap`, each
    residue's displacement is log-tapered past `cap`."""
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


def evaluate_clustering(dist, labels):
    """Silhouette score on precomputed distances, noise (-1) excluded; NaN
    if fewer than 2 clusters."""
    from sklearn.metrics import silhouette_score

    dist = np.asarray(dist)
    labels = np.asarray(labels)
    mask = labels != -1

    if len(np.unique(labels[mask])) < 2:
        print("Silhouette score: undefined (fewer than 2 non-noise clusters)")
        return float("nan")

    # silhouette_score needs an exact-zero diagonal
    sub_dist = dist[np.ix_(mask, mask)].copy()
    np.fill_diagonal(sub_dist, 0)

    score = silhouette_score(sub_dist, labels[mask], metric="precomputed")
    print(f"Silhouette score: {score:.3f}")
    return score
