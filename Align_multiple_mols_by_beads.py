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

    out = minimize( residual_worker, fit_params,
                   args=(beads_1,beads_2,rep1,rep2),
                    method='nelder' )

    aligned_beads_2 = transform(beads_2,out.params['p0'],out.params['p1'],out.params['x'],out.params['y'],out.params['z'])

    final_res = residual_worker(fit_params,beads_1,aligned_beads_2,rep1,rep2 )

    return aligned_beads_2 , final_res

### choose the first mol as the reference

reference_beads = total_beads[0][0]

reference_rep = total_reps[0][0]
'''
total_beads = total_beads[1:]
total_reps = total_reps[1:]


total_aligned_beads = []
total_aligned_reps = []

mol_count = 0

for prob_beads, prob_rep in  zip(total_beads,total_reps):

    print("mol count", mol_count)

    residuals = []

    aligned_beads_list = []

    conf_ids_ = []

    coutner = 0

    for prob_conf_beads , prob_conf_rep in tqdm.tqdm(zip( prob_beads , prob_rep  )):

        temp_align_beads , residual = allign_reps(reference_beads,prob_conf_beads,reference_rep,prob_conf_rep)

        residuals.append(residual)

        aligned_beads_list.append( temp_align_beads )

        conf_ids_.append(coutner)

    total_aligned_beads.append( aligned_beads_list[np.argmin(residuals)] )

    total_aligned_reps.append( prob_rep[np.argmin(residuals)] )

    mol_count+=1

pickle.dump(total_aligned_beads,open("total_aligned_beads.p","wb"))
pickle.dump(total_aligned_reps,open("total_aligned_reps.p","wb"))
'''

total_aligned_beads = pickle.load(open("total_aligned_beads.p","rb"))
total_aligned_reps = pickle.load(open("total_aligned_reps.p","rb"))

####draw a picture of these beads

xyz = open("aligned_beads.xyz","w")

total = 0

max_beads = 0

for beads in total_aligned_beads:

    total+=len(beads)

    if len(beads) > max_beads:

        max_beads += (len(beads) - max_beads)

print(max_beads)

for b in reference_beads:

    total+=1


xyz.write(str(total) + "\n" + "\n")

all_aligned_beads = []

for beads in total_aligned_beads:

    for b in beads:

        xyz.write("O " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + "\n")

        all_aligned_beads.append(b)


for b in reference_beads:
    xyz.write("Xe " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + "\n")


xyz.close()
'''
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=max_beads, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(all_aligned_beads)

print(km.cluster_centers_)

print(reference_beads)

print(y_km)

'''

distance_matrix = np.zeros((len(all_aligned_beads),len(all_aligned_beads)))



for i, b_i in enumerate(all_aligned_beads):

    for j, b_j in enumerate(all_aligned_beads):

        if i == j:

            distance_matrix[i,j] = 0

        elif i <j:

            d = np.linalg.norm(b_i - b_j)

            distance_matrix[i,j] = d

            distance_matrix[j,i] = d
# Ensure the correlation matrix is symmetric


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.


corr = np.exp(- distance_matrix )

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, ax=ax1, leaf_rotation=90,
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()

# how many clusters do we need before the cluster labels appear in each molecule only once

cluster_ids = hierarchy.fcluster(dist_linkage, 15, criterion="distance")

c_count = 0

for beads in  total_aligned_beads:

    c_ids = []

    for b in beads:

        c_ids.append(cluster_ids[c_count])

        c_count+=1

    print(c_ids)

    if len(c_ids) != len(set(c_ids)):

        print(c_ids)

        print("try again")

        break


print(cluster_ids)

n_clusters = max(cluster_ids)

for b , id in zip(all_aligned_beads,cluster_ids):

    plt.plot(b[0],b[1],"o", color = colors[id]  ,alpha = 0.5 )

plt.show()

###next sort all the representations the same way

sorted_reps = []

count = 0

for rep in total_aligned_reps:

    sorted_reps.append(np.zeros( ( n_clusters + 1 , 9 )  ))

    print(sorted_reps)

    for  r in rep:

        sorted_reps[-1][cluster_ids[count]] = r

        count +=1

print(sorted_reps)


