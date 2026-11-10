import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys
    import os
    import pandas as pd

    sys.path.append("/scratch/mgilioli/vis/scripts/")
    from helper import read_ipf

    topology = "/scratch/mgilioli/phd_maurizio_bussilab/simulations_and_analysis/12-metainference/2-ensemble/conf_solvent_aligned.pdb"
    trajectory = "/scratch/mgilioli/phd_maurizio_bussilab/simulations_and_analysis/12-metainference/2-ensemble/traj-all-solvent-pbc-align.xtc"
    # %todo i want the script to be able to work even if the user provides only a directory of pdbs, given that the topology is valid for all of them
    # dirframes
    interaction_file = "/scratch/mgilioli/PHD_Maurizio_BussiLab/simulations_and_analysis/12-3-analyze_interactions/rna_interface_contacts/rna_interface_contacts.csv"


    # check presence of files
    if not os.path.exists(interaction_file):
        raise Exception(f"{interaction_file} does not exist unfortunately")


    # read the interaction file
    tab = read_ipf(interaction_file)
    df = pd.DataFrame.from_dict(tab)

    # make groups

    return


@app.cell
def _():
    # Verify that all the lists have same length

    return


if __name__ == "__main__":
    app.run()
