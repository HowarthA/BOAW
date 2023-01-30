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
import pickle
import copy
from rdkit.Chem import Descriptors3D
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.optimize import linear_sum_assignment
from rdkit.Chem import Draw
from sklearn.metrics import pairwise_distances

colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.4, 0.74),
          (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]

def write_mol_xyz(mol, kmeans,title):
    f = open(title+".xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), mol.GetConformer(0).GetPositions()):

        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()

def normalise_rep(reps, stdev, means):

    copy_reps = []

    for i, rep in enumerate(reps):

        copy_reps.append([])

        for j, rep_c in enumerate(rep):
            copy_reps[-1].append((rep_c - means) / stdev)

    return copy_reps

R = 2 * 2.1943998623787615

mols = np.array(pickle.load(open("mols_cb.p", "rb")))

img=Draw.MolsToGridImage(mols[[0,1]])

img.save('mols_0_1.png')

total_beads = pickle.load(open("total_beads.p","rb"))

total_reps = pickle.load(open("total_reps.p","rb"))

all_reps = []

for rep in total_reps:

    if len(all_reps) > 0:

        for r in rep:

            all_reps = np.vstack((all_reps,r))

    else:

        all_reps = rep[0]

        for r in rep[1:]:

            all_reps = np.vstack((all_reps,r))

stds = np.std(all_reps,axis=0)
means = np.mean(all_reps,axis=0)

total_reps = normalise_rep(total_reps,stds , means)

beads_1 = total_beads[0][0]


beads_2 = total_beads[6][0]


rep1 = total_reps[0][0]


rep2 = total_reps[6][0]


print(beads_1)

write_mol_xyz(mols[0] , beads_1 , "mol0")
write_mol_xyz(mols[4] , beads_2 , "mol1")


def write_xyz(rep1, rep2):
    f = open("rep_align.xyz", "w")

    f.write(str(len(rep1) + len(rep2)) + "\n" + "\n")

    for coords in rep1:
        f.write(str("C") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in rep2:
        f.write(str("N") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()



def transform(rep, p0,p1,x,y,z):

    rep_ = copy.copy(rep)

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #find centre of mass of the second rep

    origin = np.average(rep_, axis=0)

    #convert bead positions into direction vectors from centre of mass

    rep_ -= origin

    #next rotate vectors to centre of mass by the fitted angles

    r0_ab = Rotation.from_rotvec(  -1 *basis[1] * p0 )
    r1_ab = Rotation.from_rotvec(  -1 *basis[2] * p1)

    rep_ = r0_ab.apply(rep_)

    rep_ = r1_ab.apply(rep_)

    rep_ += origin

    #next perform translations

    rep_ += np.array([ x,y,z ])

    return rep_

def residual_worker(params,coords_1,coords_2,rep1,rep2):

    #next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    x = params['x']
    y = params['y']
    z = params['z']

    coords_2 = transform(coords_2,p0,p1,x,y,z)

    #calculate the overlap of the property gaussians

    #calculate distance matrix

    D = pairwise_distances(coords_1, coords_2)

    weights = 1 / (1 +  np.exp( (D - R/2)))

    residual = np.sum(abs(rep1)) + np.sum(abs(rep2))

    for i, b1 in enumerate(rep1):

        for j, b2 in enumerate(rep2):

            if i <= j:

                #residual -= np.sum( weights[i,j]*( abs(b1 + b2)  -  abs(b1 - b2)))
                residual -= np.sum(weights[i, j] * (abs(b1 + b2) - 0.5 * abs(b1 - b2)  ))

    return   residual

def allign_reps(beads_1,beads_2,rep1,rep2):

    #print([ ds[c, r ] for c,r in zip(col_ind,row_ind)  ])

    fit_params = Parameters()

    fit_params.add("p0", value=0, min=0, max= 2 * np.pi, vary=True)
    fit_params.add("p1", value=0, min= 0, max=2 * np.pi, vary=True)

    fit_params.add("x", value=0,  vary=True)
    fit_params.add("y", value=0, vary=True)
    fit_params.add("z", value=0, vary=True)

    print(residual_worker(fit_params,beads_1,beads_2,rep1,rep2 ))



    out = minimize( residual_worker, fit_params,
                   args=(beads_1,beads_2,rep1,rep2),
                    method='nelder' )


    print(fit_report(out))

    aligned_beads_2 = transform(beads_2,out.params['p0'],out.params['p1'],out.params['x'],out.params['y'],out.params['z'])

    print(residual_worker(fit_params,beads_1,aligned_beads_2,rep1,rep2 ))

    write_xyz(beads_1,aligned_beads_2)

    return aligned_beads_2


allign_reps(beads_1,beads_2,rep1,rep2)