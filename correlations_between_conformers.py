import uuid

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
from sklearn.preprocessing import StandardScaler
from genConf import genConf

colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.4, 0.74),
          (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def write_mol_xyz(mol, kmeans, title):
    f = open(title + ".xyz", "w")

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
            copy_reps[-1].append((rep_c - means) / (stdev + 0.000001))

    return copy_reps


def embed_mol_smiles(s):
    mol = Chem.MolFromSmiles(s)
    mol = standardize(mol)

    if mol:

        mol, confIDs, energies = genConf(mol)

        return (mol, confIDs)

    else:

        return None

def embed_mol(s):

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

        confIDs = AllChem.EmbedMultipleConfs(mol, 10)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None


def make_mol(mol, confIDs):
    all_coords = []
    all_masses = []
    all_aromatic = []
    all_atom_number = []

    for id in confIDs:

        positions = mol.GetConformer(id).GetPositions()

        for atom, p in zip(mol.GetAtoms(), positions):
            all_masses.append(atom.GetMass())
            all_atom_number.append(atom.GetAtomicNum())
            all_aromatic.append(atom.GetIsAromatic())
            all_coords.append(p)

    return np.array(all_masses), np.array(all_coords), np.array(all_atom_number), np.array(all_aromatic)


def find_basis(coordinates, masses):
    origin = np.average(coordinates, axis=0, weights=masses)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return direction_vector, origin


def change_basis(m, confs, direction_vector, origin):
    for c in confs:

        coordinates = m.GetConformer(c).GetPositions()

        a_c = []

        for a_ in coordinates:
            a_ = np.matmul(direction_vector, a_)

            a_c.append(a_)

        atom_coords = np.array(a_c)

        atom_coords -= origin

        conf = m.GetConformer(c)

        for i in range(m.GetNumAtoms()):
            x, y, z = atom_coords[i]

            conf.SetAtomPosition(i, Point3D(x, y, z))

    return m


def residual2(params, atomic_nums, unaccounted_atom_pos, beads, dist, connection_bead_position):
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # get initial vectors along z principle axis

    initial_vector = copy.copy(basis[0])

    # next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    r0_ab = Rotation.from_rotvec(-1 * basis[1] * p0)
    r1_ab = Rotation.from_rotvec(-1 * basis[2] * p1)

    initial_vector = r0_ab.apply(initial_vector)

    initial_vector = r1_ab.apply(initial_vector)

    new_bead_location = dist * initial_vector + connection_bead_position

    ############next work out how many of the unnaccounted for atoms this bead now accounts for

    ############

    dists = np.array([np.linalg.norm(new_bead_location - p) for p in unaccounted_atom_pos])

    scores = 1 / (1 + np.exp(-2 * (dists - dist / 2))) * atomic_nums

    return np.sum(scores)


def match_to_substructures(mol):
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    # changes log for HAcceptorSmarts:
    #  v2, 1-Nov-2008, GL : fix amide-N exclusion; remove Fs from definition

    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0])]')

    # find any atoms that match these macs keys

    donor_atoms = [i[0] for i in mol.GetSubstructMatches(HDonorSmarts, uniquify=1)]

    acceptor_atoms = [i[0] for i in mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1)]

    is_donor_atom = []

    is_acceptor_atom = []

    for atom in mol.GetAtoms():

        id = atom.GetIdx()

        if id in donor_atoms:

            is_donor_atom.append(True)

        else:
            is_donor_atom.append(False)

        if id in acceptor_atoms:

            is_acceptor_atom.append(True)

        else:

            is_acceptor_atom.append(False)

    return is_donor_atom, is_acceptor_atom


def make_representation(total_beads, m, bead_dist, confids):
    reps = []

    for i, beads in zip(confids, total_beads):

        atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m, [i])

        #####

        is_donor_atom, is_acceptor_atom = match_to_substructures(m)

        ############# next job is to "color the beads in"

        # define some atomwise properties

        ComputeGasteigerCharges(m)

        charges = []

        for a in m.GetAtoms():
            charges.append(float(a.GetProp("_GasteigerCharge")))

        charges = np.array(charges)

        CC = np.array(rdMolDescriptors._CalcCrippenContribs(m))

        ASA = np.array([k for k in rdMolDescriptors._CalcLabuteASAContribs(m)[0]])

        TPSA = np.array(rdMolDescriptors._CalcTPSAContribs(m))

        logP_c = CC[:, 0]

        MR_c = CC[:, 1]

        ###### next make the representation

        representation = np.zeros((len(beads), 9))

        # find distances to atoms from beads

        bead_dists_to_atoms = np.array([np.linalg.norm(b - atom_coords, axis=1) for b in beads])

        # find normalised vectors to atoms from beads

        # vectors_to_atoms = np.array([b - atom_coords for b in beads])
        # vectors_to_atoms /= np.linalg.norm(vectors_to_atoms, axis=2)[:, :, None]

        ind1 = 0

        for b, ds in zip(beads, bead_dists_to_atoms):
            weights = 1 / (1 + np.exp(2 * (ds - bead_dist / 2)))

            # weights = 1 / (1 + np.exp(  (ds -  bead_dist)))
            # make a vector of charges

            charge_vector = np.sum(weights * charges)
            representation[ind1, 0] = charge_vector

            # could have a dipole vector too
            # dipole_vectors = np.sum(weights * charges * dist,axis = 1)
            # mass vectors

            mass_vectors = np.sum(weights * atom_masses)

            representation[ind1, 1] = mass_vectors

            # logP vectors

            logP_vectors = np.sum(weights * logP_c)

            representation[ind1, 2] = logP_vectors

            # MR vectors - steric descriptors

            MR_vector = np.sum(weights * MR_c)

            representation[ind1, 3] = MR_vector

            # ASA - surface area descriptors

            ASA_vector = np.sum(weights * ASA)

            representation[ind1, 4] = ASA_vector

            # TPSA - surface area descriptors

            TPSA_vector = np.sum(weights * TPSA)

            representation[ind1, 5] = TPSA_vector

            # aromatic

            Aromatic_vector = np.sum(weights * atom_aromatic)

            representation[ind1, 6] = Aromatic_vector

            # HDB

            HDB_vector = np.sum(weights * is_donor_atom)

            representation[ind1, 7] = HDB_vector

            # HBA

            HDA_vector = np.sum(weights * is_acceptor_atom)

            representation[ind1, 8] = HDA_vector

            ind1 += 1

        reps.append(representation)

    return np.array(reps)


def make_beads(m, confIDs, dist, counter):
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    t_beads = []

    #### rotate mol onto PC axis

    for i in confIDs:

        atom_masses, atom_coords, atomic_nums, all_aromatic = make_mol(m, [i])

        # atom_masses \= np.sum(atom_masses)

        origin = np.average(atom_coords, weights=atom_masses, axis=0)

        closest_atom = np.argmin(np.linalg.norm(origin - atom_coords, axis=1))

        beads = np.array([atom_coords[closest_atom]])

        ###

        #####

        # build to model by growing a web around the molecule

        # place initial bead

        # need to count atoms that are accounted for by this bead

        atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in atom_coords])

        unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist

        unaccounted_atom_pos = atom_coords[unaccounted]

        unaccounted_atom_pos_old = len(unaccounted_atom_pos) + 1

        unnaccounted_atom_nums = atomic_nums[unaccounted]

        bead_n = 0

        while len(unaccounted_atom_pos) > 0 & (len(unaccounted_atom_pos) < unaccounted_atom_pos_old):

            # find which bead has the most atoms around it in a radius of 2*R>x>R

            accounted_ns = []

            new_beads_positions = []

            for bead_for_connection, connection_bead_position in enumerate(beads):
                # select the bead that has the most

                fit_params = Parameters()

                fit_params.add("p0", value=0, min=- np.pi, max=+ np.pi, vary=True)
                fit_params.add("p1", value=0, min=- np.pi, max=+ np.pi, vary=True)

                out = minimize(residual2, fit_params,
                               args=(
                                   unnaccounted_atom_nums, unaccounted_atom_pos, beads, dist, connection_bead_position),
                               method='nelder', options={'maxiter': 10000})

                # add this bead to the total

                initial_vector = basis[0]

                # next apply interpolated rotations

                p0 = out.params['p0']
                p1 = out.params['p1']

                r0_ab = Rotation.from_rotvec(-1 * basis[1] * p0)
                r1_ab = Rotation.from_rotvec(-1 * basis[2] * p1)

                initial_vector = r0_ab.apply(initial_vector)

                initial_vector = r1_ab.apply(initial_vector)

                new_bead_location = dist * initial_vector + connection_bead_position

                # next remove atoms that are now accounted for

                atoms_dist_to_beads = np.linalg.norm(unaccounted_atom_pos - new_bead_location, axis=1)

                accounted = atoms_dist_to_beads < dist

                accounted_ns.append(np.sum(accounted))

                new_beads_positions.append(new_bead_location)

            ### decide on which connection lead to the greatest increase in new atoms accounted for

            # if (len(beads) > 2) & (np.all(unnaccounted_ns == unnaccounted_ns[0])):

            #   break

            best = accounted_ns.index(max(accounted_ns))

            new_bead = new_beads_positions[best]

            beads = np.vstack((beads, new_bead))

            atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in atom_coords])

            unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist

            unaccounted_atom_pos_old = len(unaccounted_atom_pos)

            unaccounted_atom_pos = atom_coords[unaccounted]

            unnaccounted_atom_nums = atomic_nums[unaccounted]

            bead_n += 1

        if i == 0:
            write_mol_xyz(m, beads, "mol_" + str(counter))

        t_beads.append(beads)

    return t_beads

R = 2 * 2.1943998623787615

property = 'b_ratio'

df = pd.read_csv("/Users/alexanderhowarth/Desktop/total_b_ratio.csv").dropna(subset=property)

df = df.drop_duplicates(subset='Compound_ID')

code = "TRB0005601 series"

df = df[df["STRUCTURE_COMMENT"] == code]

IC50 = []
smiles = []

for i, r in tqdm.tqdm([(i, r) for i, r in df.iterrows()]):

    print(r['Compound_ID'])

    if r[property] > 0.9:

        IC50.append(r[property])

        smiles.append(r['Smiles'])

print(IC50)
print(smiles)

mols = []

confs = []

for s in tqdm.tqdm(smiles):
    mol, confIDs = embed_mol(s)

    mols.append(mol)

    confs.append([i for i in confIDs])

for i, m in enumerate(mols):
    all_masses, all_coords, all_atom_number, all_aromatic = make_mol(m, confs[i])

    direction_vector, origin = find_basis(all_coords, all_masses)

    change_basis(m, confs[i], direction_vector, origin)

# next make beads and all representations for all conformers and mols

total_beads = []

total_reps = []

for i, m in tqdm.tqdm(enumerate(mols)):

    beads = make_beads(m, confs[i], R, i)

    rep = make_representation(beads, m, R, confs[i])

    total_beads.append(beads)
    total_reps.append(rep)

all_reps = []

for rep in total_reps:

    if len(all_reps) > 0:

        for r in rep:
            all_reps = np.vstack((all_reps, r))

    else:

        all_reps = rep[0]

        for r in rep[1:]:

            all_reps = np.vstack((all_reps, r))


stds = np.std(all_reps, axis=0)
means = np.mean(all_reps, axis=0)

print(stds)

total_reps = normalise_rep(total_reps, stds, means)


def write_xyz(rep1, rep2):
    f = open("tes_folder/rep_align"+str(uuid.uuid4() )+ ".xyz", "w")

    f.write(str(len(rep1) + len(rep2)) + "\n" + "\n")

    for coords in rep1:
        f.write(str("C") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in rep2:
        f.write(str("N") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def transform(rep, p0, p1, x, y, z):
    rep_ = copy.copy(rep)

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # find centre of mass of the second rep

    origin = np.average(rep_, axis=0)

    # convert bead positions into direction vectors from centre of mass

    rep_ -= origin

    # next rotate vectors to centre of mass by the fitted angles

    r0_ab = Rotation.from_rotvec(-1 * basis[1] * p0)
    r1_ab = Rotation.from_rotvec(-1 * basis[2] * p1)

    rep_ = r0_ab.apply(rep_)

    rep_ = r1_ab.apply(rep_)

    rep_ += origin

    # next perform translations

    rep_ += np.array([x, y, z])

    return rep_


def align_residual(params, coords_1, coords_2, rep1, rep2):
    # next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    x = params['x']
    y = params['y']
    z = params['z']

    coords_2 = transform(coords_2, p0, p1, x, y, z)

    # calculate the overlap of the property gaussians

    # calculate distance matrix

    D = pairwise_distances(coords_1, coords_2)

    weights = 1 / (1 + np.exp((D - R / 2)))

    norm = np.sum(abs(rep1)) + np.sum(abs(rep2))

    residual = 0

    for i, b1 in enumerate(rep1):

        for j, b2 in enumerate(rep2):

            if i <= j:
                # residual -= np.sum( weights[i,j]*( abs(b1 + b2)  -  abs(b1 - b2)))
                residual -= np.sum(weights[i, j] * (abs(b1 + b2) - 0.5 * abs(b1 - b2)))

    residual = norm + residual

    return residual


def allign_reps(beads_1, beads_2, rep1, rep2):
    # print([ ds[c, r ] for c,r in zip(col_ind,row_ind)  ])

    fit_params = Parameters()

    fit_params.add("p0", value=0, min=0, max=2 * np.pi, vary=True)
    fit_params.add("p1", value=0, min=0, max=2 * np.pi, vary=True)

    fit_params.add("x", value=0, vary=True)
    fit_params.add("y", value=0, vary=True)
    fit_params.add("z", value=0, vary=True)

    out = minimize(align_residual, fit_params,
                   args=(beads_1, beads_2, rep1, rep2),
                   method='nelder')

    initial_res = align_residual(fit_params, beads_1, beads_2, rep1, rep2)

    aligned_beads_2 = transform(beads_2, out.params['p0'], out.params['p1'], out.params['x'], out.params['y'],
                                out.params['z'])

    final_res = align_residual(fit_params, beads_1, aligned_beads_2, rep1, rep2)

    final_res = ( initial_res - final_res ) / initial_res

    print(final_res)

    return aligned_beads_2, final_res


def train_RF(train_descs, test_descs, train_IC50, test_IC50):
    train_fps = np.array([i for i in train_descs])

    test_fps = np.array(test_descs)

    train_y = np.array(train_IC50)

    test_y = np.array(test_IC50)

    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)[0]

    return test_pred, test_y


### choose the first mol as the reference

leng =  0

for i_beads, i_rep in zip(total_beads, total_reps):

    for i_conf_beads, i_conf_rep in zip(i_beads, i_rep):

        leng+=1

residuals = np.zeros((leng,leng))

i =0

for i_beads, i_rep in zip(total_beads, total_reps):

    for i_conf_beads, i_conf_rep in zip(i_beads, i_rep):

        j = 0

        for j_beads, j_rep in zip(total_beads, total_reps):

            for j_conf_beads, j_conf_rep in zip(j_beads, j_rep):

                if i == j:

                    residuals[i,j] = 0
                    residuals[j, i] = 0

                if i < j:

                    temp_align_beads, residual = allign_reps(i_conf_beads, j_conf_beads,i_conf_rep,
                                                             j_conf_rep)

                    residuals[i,j] = residual
                    residuals[j, i] = residual

                j+=1

        i+=1

pickle.dump(residuals,open("residuals.p","wb"))

distance_matrix = pickle.load(open("residuals.p","rb"))

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



