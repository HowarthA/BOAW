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
import os
from rdkit.Chem import Descriptors
colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.4, 0.74),
          (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]

#df = pd.DataFrame(pickle.load(open( os.path.expanduser( "~/binding_data.p"), "rb"))).reset_index()

df = pd.DataFrame(pickle.load(open( "../binding_data.p", "rb"))).reset_index()



R = 2 * 2.1943998623787615

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

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

def embed_mol_smiles(s):

    mol = rdmolops.AddHs(s)

    if mol:

        rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

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

        mol = rdmolops.RemoveHs(mol)

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

    initial_vector = basis[0]

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

def make_representation(total_beads, m, bead_dist,confids):

    reps = []

    for i , beads in zip(confids , total_beads):

        atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m,[i])

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

        '''
        for b, ds, v in zip(beads, bead_dists_to_atoms, vectors_to_atoms):
            weights = np.exp(-1 * ( ds / bead_dist) ** 2)

            # make a vector of charges
            charge_vector = np.sum((weights * charges)[:, None] * v, axis=0)

            representation[ind1, 0, :] = charge_vector

            # could have a dipole vector too
            # dipole_vectors = np.sum(weights * charges * dist,axis = 1)

            # mass vectors
            mass_vectors = np.sum((weights * atom_masses)[:, None] * v, axis=0)

            representation[ind1, 1, :] = mass_vectors

            # logP vectors

            logP_vectors = np.sum((weights * logP_c)[:, None] * v, axis=0)

            representation[ind1, 2, :] = logP_vectors

            # MR vectors - steric descriptors

            MR_vector = np.sum((weights * MR_c)[:, None], axis=0)

            representation[ind1, 3, :] = MR_vector

            # ASA - surface area descriptors

            ASA_vector = np.sum((weights * ASA)[:, None], axis=0)

            representation[ind1, 4, :] = ASA_vector

            # TPSA - surface area descriptors

            TPSA_vector = np.sum((weights * TPSA)[:, None], axis=0)

            representation[ind1, 5, :] = TPSA_vector

            ind1 += 1
        '''

        for b, ds in zip(beads, bead_dists_to_atoms):

            weights = 1 / (1 + np.exp(2 * (ds - bead_dist/2)))

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

        origin = np.average(atom_coords,weights=atom_masses ,axis=0)

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

                initial_vector = copy.copy(basis[0])

                # next apply interpolated rotations

                p0 = out.params['p0']
                p1 = out.params['p1']

                r0_ab = Rotation.from_rotvec(-1 * basis[1] * p0)
                r1_ab = Rotation.from_rotvec(-1 * basis[2] * p1)

                initial_vector = r0_ab.apply(initial_vector)

                initial_vector = r1_ab.apply(initial_vector)

                new_bead_location = dist * initial_vector + connection_bead_position

                # next remove atoms that are now accounted for

                atoms_dist_to_beads = np.linalg.norm( unaccounted_atom_pos - new_bead_location, axis=1)

                accounted = atoms_dist_to_beads < dist

                accounted_ns.append(np.sum(accounted))

                new_beads_positions.append(new_bead_location)

            ### decide on which connection lead to the greatest increase in new atoms accounted for

            #if (len(beads) > 2) & (np.all(unnaccounted_ns == unnaccounted_ns[0])):

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

        #write_mol_xyz(m , beads,"mol_" + str(counter) + ".xyz")

        t_beads.append(beads)

    return t_beads

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


def train_RF(train,test):

    train_fps = np.array([i for i in train['SOAPs']])

    test_fps = np.array([test['SOAPs']])

    train_y = np.array(train['IC50'])

    test_y = np.array(test['IC50'])

    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    rf.fit( train_fps,train_y)

    test_pred = rf.predict(test_fps)[0]

    return test_pred, test_y


def AlignRepWorker(mols, confs,r_id):



    for i, m in enumerate(mols):

        all_masses, all_coords, all_atom_number, all_aromatic = make_mol(m, confs[i])

        direction_vector, origin = find_basis(all_coords, all_masses)

        change_basis(m, confs[i], direction_vector, origin)

    # next make beads and all representations for all conformers and mols

    total_beads = []

    total_reps = []

    for i, m in enumerate(mols):

        beads = make_beads(m, confs[i], R, i)

        rep = make_representation(beads, m, R, confs[i])

        total_beads.append(beads)
        total_reps.append(rep)

    # mols = np.array(pickle.load(open("mols_cb.p", "rb")))
    # total_beads = pickle.load(open("total_beads.p","rb"))
    # total_reps = pickle.load(open("total_reps.p","rb"))

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

    total_reps = normalise_rep(total_reps, stds, means)

    ### choose the first mol as the reference

    reference_beads = total_beads[0][0]

    reference_rep = total_reps[0][0]

    total_beads = total_beads[1:]
    total_reps = total_reps[1:]

    total_aligned_beads = []
    total_aligned_reps = []

    mol_count = 0

    for prob_beads, prob_rep in zip(total_beads, total_reps):

        residuals = []

        aligned_beads_list = []

        conf_ids_ = []

        coutner = 0

        for prob_conf_beads, prob_conf_rep in zip(prob_beads, prob_rep):

            temp_align_beads, residual = allign_reps(reference_beads, prob_conf_beads, reference_rep, prob_conf_rep)

            residuals.append(residual)

            aligned_beads_list.append(temp_align_beads)

            conf_ids_.append(coutner)

        total_aligned_beads.append(aligned_beads_list[np.argmin(residuals)])

        total_aligned_reps.append(prob_rep[np.argmin(residuals)])

        mol_count += 1

    # pickle.dump(total_aligned_beads,open("total_aligned_beads.p","wb"))
    # pickle.dump(total_aligned_reps,open("total_aligned_reps.p","wb"))
    # total_aligned_beads = pickle.load(open("total_aligned_beads.p","rb"))
    # total_aligned_reps = pickle.load(open("total_aligned_reps.p","rb"))

    all_aligned_beads = []

    for b in reference_beads:
        all_aligned_beads.append(b)

    max_beads = 0

    for beads in total_aligned_beads:

        if len(beads) > max_beads:
            max_beads += (len(beads) - max_beads)

        for b in beads:
            all_aligned_beads.append(b)

    total_aligned_reps = [reference_rep] + total_aligned_reps

    total_aligned_beads = [reference_beads] + total_aligned_beads

    distance_matrix = np.zeros((len(all_aligned_beads), len(all_aligned_beads)))

    for i, b_i in enumerate(all_aligned_beads):

        for j, b_j in enumerate(all_aligned_beads):

            if i == j:

                distance_matrix[i, j] = 0

            elif i < j:

                d = np.linalg.norm(b_i - b_j)

                distance_matrix[i, j] = d

                distance_matrix[j, i] = d

    # Ensure the correlation matrix is symmetric

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.

    corr = np.exp(- distance_matrix)

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

    plt.savefig( "clusters/clusters_" + r_id + ".png")

    plt.close()

    # how many clusters do we need before the cluster labels appear in each molecule only once

    #cluster_ids = hierarchy.fcluster(dist_linkage, 15, criterion="distance")

    cluster_ids = hierarchy.fcluster(dist_linkage, max_beads, 'maxclust')

    c_count = 0

    split_c_ids = []

    for beads in total_aligned_beads:

        c_ids = []

        for b in beads:

            c_ids.append(cluster_ids[c_count])

            c_count += 1

        split_c_ids.append(c_ids)

    n_clusters = max(cluster_ids) + 1

    for b, id in zip(all_aligned_beads, cluster_ids):
        plt.plot(b[0], b[1], "o", color=colors[id], alpha=0.5)

    plt.savefig("clusters/beads_" + r_id + ".png")

    plt.close()

    ###next sort all the representations the same way

    sorted_reps = []

    count = 0

    for rep, c_ids in zip(total_aligned_reps,split_c_ids):

        sorted_reps.append(np.zeros((n_clusters , 9)))

        for c in range(0,n_clusters ):

            w = np.where(np.array(c_ids) == c )[0]

            if len(w) > 0:

                sorted_reps[-1][c] = np.mean(rep[w],axis=0)

            count += 1

    sorted_reps = np.array(sorted_reps)

    sorted_reps = [s.flatten() for s in sorted_reps]

    return sorted_reps


def compare_same_dataset(mols):

    fps = [ ]

    to_remove = []

    for m in mols:

        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048))

    maxes = []

    i = 0

    for fp in fps:

        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)

        sims[i] = 0

        if max(sims) < 0.6:

            to_remove.append(i)

        else:

            maxes.append(np.mean(np.sort(sims)[::-1][0:min([10, len(fps)])]))

        i+=1

    m_sim = np.median(maxes)

    return to_remove,m_sim


def rep(m):

    rep_ = [

        #graph reps

        #2d desc
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        rdMolDescriptors.CalcLabuteASA(m),
        rdMolDescriptors.CalcCrippenDescriptors(m)[1],
        rdMolDescriptors.CalcTPSA(m),

        #rdMolDescriptors.CalcNumAliphaticCarbocycles(m),
        #rdMolDescriptors.CalcNumAliphaticHeterocycles(m),
        #rdMolDescriptors.CalcNumAliphaticRings(m),
        #rdMolDescriptors.CalcNumAromaticCarbocycles(m),
        #rdMolDescriptors.CalcNumAromaticHeterocycles(m),
        rdMolDescriptors.CalcNumAromaticRings(m),

        rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumHBD(m),

        #rdMolDescriptors.CalcNumHeteroatoms(m),
        #rdMolDescriptors.CalcNumHeterocycles(m),
        #rdMolDescriptors.CalcNumRings(m),
        #rdMolDescriptors.CalcNumRotatableBonds(m),

        ]

    return [float(r) for r in rep_]

def physchem_KRR(args):

    ind1 = args[0]

    r = args[1]

    data_size_cut_off = 20

    max_data_cut_off = 300

    IC50 = []
    mols_ = []

    for v, m in zip(r['IC50 (nM)'], r['Mols']):

        v = str(v)

        v = v.replace(" ", "")

        if (">" not in v) and ("<" not in v):

            v = float(v)

            if not np.isnan(v):

                IC50.append(np.log(v))

                mols_.append(m)

    to_remove,m_sim = compare_same_dataset(mols_)

    mols_ = np.delete(np.array(mols_) , to_remove)

    IC50 = np.delete(np.array(IC50),to_remove)

    if (len(IC50) > data_size_cut_off) and (len(IC50) < max_data_cut_off):

        descs = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols_ ]

        df_soap = pd.DataFrame([[S, Ic] for S, Ic in zip(descs, IC50)], columns=["SOAPs", "IC50"])

        av_vs = []

        av_ps = []

        for i2, r_ in df_soap.iterrows():

            test = r_

            train = df_soap.drop(i2)

            # make predictions

            test_pred, test_val = train_RF(train, test )

            av_vs.append(test_val)

            av_ps.append(test_pred)

        r2 = r2_score(av_vs, av_ps)

        rmse_med = mean_squared_error(av_vs, [np.median(av_vs) for j in av_vs])

        rmse = mean_squared_error(av_vs, av_ps)

        std_errors = np.std([abs(v - p) for v, p in zip(av_vs, av_ps)])

        plt.plot(av_vs, av_ps, "o", label="R2 = " + str(r2) + "\nstd = " + str(std_errors) + "\nRMSE = " + str(rmse) + "\nRMSE/RMSE_No_skill = " + str(rmse/rmse_med))
        plt.plot([min(av_vs), max(av_vs)], [min(av_vs), max(av_vs)], linestyle=":")

        plt.legend()

        plt.savefig("plots/beads_" + r['ID'] + ".png")

        plt.close()

        res = [[r['ID'], len(mols_), m_sim, r2, rmse, std_errors,rmse/rmse_med]]

        res_df = pd.DataFrame(res,
                          columns=['file', 'length', 'mean_sim', 'R2_of_means', 'RMSE', 'std_errors','RMSE_No_skill'])

        res_df.to_csv("results/binding_" + r['ID'] + ".csv")

        print("finished" , r['ID'])

################################## mac multiprocessing

args = [ ( i, r ) for i,r in df.iterrows() ]

##################divide into equal pieces
'''
for a in args:

    physchem_KRR(a)

'''


import multiprocessing

p = multiprocessing.Pool(processes=60)

# defaults to os.cpu_count() workers

p.map_async(physchem_KRR, args)

# perform process for each i in i_list

p.close()
p.join()
