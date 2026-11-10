"""Residue labels of the contact file, e.g. 'ARG152.I' or 'C4.C'.

No dependencies, so both the pipeline and the recovery tool (which runs in a
plain PyMOL install) use the same parser.
"""
import re

LABEL_RE = re.compile(r"([A-Za-z0-9]+?)(-?\d+)\.(\w+)")


def parse_label(label):
    """'ARG152.I' -> ('ARG', 152, 'I'); 'C4.C' -> ('C', 4, 'C')."""
    match = LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Cannot parse residue label {label!r}")
    resname, resid, chain = match.groups()
    return resname, int(resid), chain


def residue_selection(label, obj):
    """PyMOL selection of residue `label` in object `obj`."""
    resname, resid, chain = parse_label(label)
    return f"({obj} and chain {chain} and resi \\{resid} and resn {resname})"
