from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from lmfit import Parameters, fit_report, minimize
from scipy.spatial.transform import Rotation
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import rdMolAlign
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem import AllChem
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import r2_score
from rdkit import DataStructs
from sklearn.metrics import mean_squared_error
from rdkit.Chem import rdMolDescriptors
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D
import os

#####

database = os.path.expanduser("/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi")

####calculate fps

db_length = len(open(database, "r").readlines())

print(db_length)

### select diverse set of n = 100000

fps = []

smiles = []

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

i = 0

for s in tqdm.tqdm(open(database, "r").readlines()):

    s = s.split()[0]

    try:
        m = Chem.MolFromSmiles(s)

        m = standardize(m)

        fps.append(AllChem.GetMorganFingerprintAsBitVect(m,2,1024))

        smiles.append(Chem.MolToSmiles(m))

    except:

        print("broken mol")
