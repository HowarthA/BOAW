from rdkit.Chem import SDMolSupplier
from rdkit.Chem import SDWriter
import numpy as np

import pandas as pd
import os

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import random

rdDepictor.SetPreferCoordGen(True)
import numpy as np
from rdkit.Chem import AllChem
from rdkit import RDLogger
from matplotlib import pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
import tqdm
from scipy.stats import gaussian_kde
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
import pickle

folder = "/Users/alexanderhowarth/Desktop/5061_NN_search/"


NNs = []

for f in os.listdir(folder):

    if f.endswith(".sdf"):

        print(f)

        try:

            for m in SDMolSupplier(folder + f):

                NNs.append(m)

        except:

            print("bad file" , f)

print(len(NNs))

#write file

with SDWriter(folder + "/NNs_.sdf") as w:

   for m1 in NNs:

        w.write(m1)

lp_minmax = rdSimDivPickers.MaxMinPicker()
lp_leader = rdSimDivPickers.LeaderPicker()


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def remove_duplicates_by_inchi_key(smiles_list):
    uncharger = rdMolStandardize.Uncharger()
    te = rdMolStandardize.TautomerEnumerator()

    Mols = []
    inchis = []

    for s in smiles_list:

        m = Chem.MolFromSmiles(s)

        if m:
            m = standardize(m)

            clean_mol = rdMolStandardize.Cleanup(m)
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            m = te.Canonicalize(uncharged_parent_clean_mol)

            i = Chem.MolToInchiKey(m)

            if i not in inchis:
                Mols.append(m)

    return Mols


def compare_different_dataset(fps1, fps2):
    x = np.linspace(0, 1, 1000)

    maxes = np.zeros(len(fps1))

    i = 0

    for fp in tqdm.tqdm(fps1):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps2)

        maxes[i] = max(sims)

        i += 1

    kde = gaussian_kde(maxes).pdf(x)

    return x, kde


def compare_different_dataset_images(fps1, fps2, mols_1, mols_2):
    x = np.linspace(0, 1, 1000)

    maxes = np.zeros(len(fps1))

    i = 0

    for fp in tqdm.tqdm(fps1):

        sims = DataStructs.BulkTanimotoSimilarity(fp, fps2)

        maxes[i] = max(sims)

        if max(sims) > 0.99:
            img = Draw.MolsToGridImage([mols_1[i], mols_2[np.argmax(np.array(sims))]])

            img.save("images/" + "Duplicates" + str(random.randint(1000, 9999)) + ".png")

        i += 1

    kde = gaussian_kde(maxes).pdf(x)

    return x, kde


def compare_same_dataset(centre_fp, fps):

    x = np.linspace(0, 1, 1000)

    i = 0

    sims = DataStructs.BulkTanimotoSimilarity(centre_fp, fps)

    kde = gaussian_kde(sims).pdf(x)

    return x, kde, sims

input_file = folder + "/NNs_.sdf"

hit_smiles = "O=C(C1OC2=C(C=CC=C2)C=1)NS(C1[C@@H]2C[C@@H](CC2)C1)(=O)=O"

Id_code ="TRB0005061"

initial_hit = standardize(Chem.MolFromSmiles(hit_smiles))

Mols = []

boanw_sim = []

for m in Chem.SDMolSupplier(input_file):

    if m:

        try:

            boanw_sim.append(float(m.GetProp("_similarity")))

            Mols.append(standardize(m))

        except:

            None


initial_hit = standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)

fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in tqdm.tqdm(Mols)]

# compare_different_dataset_images(Sellec_fps , FDA_fps,Selleck_Mols,FDA_Mols)

x, pdf , sims = compare_same_dataset(centre_fp, fps)



plt.plot(sims,boanw_sim,"o",alpha = 0.2)

plt.xlabel("Tanimoto Similarity")
plt.ylabel("BOAW Similarity")

plt.savefig(folder + "/Compare_similarity.png", format="PNG", bbox_inches='tight')
plt.savefig(folder + "/Compare_similarity.svg", format="SVG", bbox_inches='tight')

plt.close()

plt.plot(x, pdf, label="Similarity to Initial Hit", linewidth=3)

plt.xlabel("Tanimoto Similarity")
plt.ylabel("Density")

plt.legend()

plt.savefig(folder + "/NN_distributions.png", format="PNG", bbox_inches='tight')
plt.savefig(folder + "/NN_distributions.svg", format="SVG", bbox_inches='tight')

plt.close()


########################################################################################################################
# Do some physchem plots
########################################################################################################################

def plot_property_kde(property_list):
    kde = gaussian_kde(property_list)

    x = np.linspace(min(property_list), max(property_list), 1000)

    pdf = kde.pdf(x)

    return x, pdf


def get_props_2d_continious(m):
    return [Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m), Crippen.MolLogP(m),
            rdMolDescriptors.CalcFractionCSP3(m)]


df = pd.DataFrame([get_props_2d_continious(m) for m in Mols], columns=['mw', 'TPSA', 'cLogP', 'Fraction SP3'])

original_props = get_props_2d_continious(initial_hit)

fig, axs = plt.subplots(ncols=2, nrows=2)

x, pdf = plot_property_kde(df['mw'])

axs[0, 0].plot(x, pdf, linewidth=3, color="C0", label="Generated Mols")

axs[0, 0].axvline(original_props[0], linewidth=3, color="C0", linestyle="--", label=Id_code, alpha=0.5)

axs[0, 0].set_xlabel("Molecular Weight")

axs[0, 0].set_xlim([0, 1000])

axs[0, 0].legend()

####

x, pdf = plot_property_kde(df['TPSA'])

axs[0, 1].plot(x, pdf, linewidth=3, color="C1", label="Generated Mols")
axs[0, 1].axvline(original_props[1], linewidth=3, color="C1", linestyle="--", label=Id_code, alpha=0.5)
axs[0, 1].set_xlim([0, 500])
axs[0, 1].set_xlabel("TPSA")
axs[0, 1].legend()

####

x, pdf = plot_property_kde(df['cLogP'])

axs[1, 0].plot(x, pdf, linewidth=3, color="C2", label="Generated Mols")
axs[1, 0].axvline(original_props[2], linewidth=3, color="C2", linestyle="--", label=Id_code, alpha=0.5)
axs[1, 0].set_xlabel("cLogP")
axs[1, 0].set_xlim([-10, 10])
axs[1, 0].legend()

####

x, pdf = plot_property_kde(df['Fraction SP3'])

axs[1, 1].plot(x, pdf, linewidth=3, color="C3", label="Generated Mols")
axs[1, 1].axvline(original_props[3], linewidth=3, color="C3", linestyle="--", label=Id_code, alpha=0.5)
axs[1, 1].set_xlabel("Fraction SP3")

axs[1, 1].legend()

fig.set_size_inches(12, 8)

plt.savefig(folder + "/Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig(folder + "/Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()