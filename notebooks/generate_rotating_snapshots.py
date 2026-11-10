import pymol
import numpy as np
import itertools
from pymol import cmd
# Make selection
STEP=30
selection = cmd.select("segid N and resid 100")
cmd.show(selection, "sticks")
# Make zoom (which comes automatically with centering)
cmd.zoom(selection, 20)

orientations = np.arange(0,360,STEP)
combinations = list(itertools.combinations(orientations, 2))
print(combinations)

for xang, yang in combinations:
   cmd.rotate(axis="x", angle=xang, selection="all", camera=0)
   cmd.rotate(axis="y", angle=yang, selection="all", camera=0)
   # Save the png numbered



