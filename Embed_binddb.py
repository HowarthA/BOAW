import os

import numpy as np
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem
import pandas as pd
import tqdm
import pickle
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem

df = [  ]

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

for f in tqdm.tqdm(os.listdir("/Users/alexanderhowarth/PycharmProjects/mol_datasets/Binding_sets")):

    s = f.split("_Validation")[0]

    for m in Chem.SDMolSupplier("/Users/alexanderhowarth/PycharmProjects/mol_datasets/Binding_sets/" + f):

        if m:

            try:
                IC50 = m.GetProp("IC50 (nM)")
            except:
                IC50 = None

            try:
                Ki = m.GetProp("Ki (nM)")
            except:
                Ki = None

            try:
                Kd = m.GetProp("Kd (nM)")
            except:
                Kd = None

            try:
                EC50 = m.GetProp("EC50 (nM)")
            except:
                EC50 = None

            m = standardize(m)

            df.append( [s , Chem.MolToSmiles(m),IC50,EC50,Ki,Kd  ] )

df = pd.DataFrame(df,columns=["ID","Smiles","IC50 (nM)","EC50 (nM)","Ki","Kd"])

df.to_csv("/Users/alexanderhowarth/PycharmProjects/mol_datasets/Binding_sets/binding.csv")

df = pd.read_csv("/Users/alexanderhowarth/PycharmProjects/mol_datasets/Binding_sets/binding.csv")

to_remove = []

mols = []

def embed_mol_smiles(s):

    mol = Chem.MolFromSmiles(s)
    mol = standardize(mol)
    mol = rdmolops.AddHs(mol)

    if mol:

        rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

        mol = Chem.AddHs(mol)

        # Generate conformers

        if rot_bonds < 8:

            n_conformers = 50

        elif (rot_bonds >= 8) and (rot_bonds <= 12):

            n_conformers = 200

        else:

            n_conformers = 300

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        mol = rdmolops.RemoveHs(mol)

        Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None

mols = [ ]

confs = []

for i, r in tqdm.tqdm(df.iterrows()):

    s = r['Smiles']

    if '*' in s:

        to_remove.append(i)

    else:

        mol, confIDs = embed_mol_smiles(s)

        mols.append(mol)

        confs.append([i for i in confIDs])

        if mol:

            mols.append(mol)

        else:
            to_remove.append(i)


df =df.drop(to_remove)

df['Mols'] = mols
df['Confs'] = confs


df = df.groupby('ID').agg(list)

pickle.dump(df,open("binding_data.p","wb"))


