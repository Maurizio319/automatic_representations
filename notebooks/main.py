import sys
import os
import pandas as pd

sys.path.append("scripts/")
from helper import read_ipf

topology = "/scratch/mgilioli/phd_maurizio_bussilab/simulations_and_analysis/12-metainference/2-ensemble/conf_solvent_aligned.pdb"
trajectory = "/scratch/mgilioli/phd_maurizio_bussilab/simulations_and_analysis/12-metainference/2-ensemble/traj-all-solvent-pbc-align.xtc"
# %todo i want the script to be able to work even if the user provides only a directory of pdbs, given that the topology is valid for all of them
# dirframes
interaction_file = "/scratch/mgilioli/phd_maurizio_bussilab/simulations_and_analysis/12-3-analyze_interactions/rna_interface_contacts/rna_interface_contacts.csv"


# check presence of files
if not os.path.exists(interaction_file):
    raise exception(f"{interaction_file} does not exist unfortunately")


# read the interaction file
df = read_ipf(interaction_file)
df = pd.dataframe.from_dict(df)
print(df)
print(df.columns)
print(df.shape)

# make groups
