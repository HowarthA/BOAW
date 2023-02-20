import os
import time
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

from rdkit.Chem import rdmolops
from rdkit.Geometry import Point3D

import copy
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from genConf import genConf
import pathos.multiprocessing as mp
import itertools

from morfeus import XTB
from morfeus import Dispersion
from morfeus import SASA

colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.4, 0.74),
          (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]

R = 2 * 2.1943998623787615


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def write_mol_xyz(mol, kmeans, title, cid):
    f = open(title + ".xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), mol.GetConformer(cid).GetPositions()):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def embed_mol(s):
    mol = Chem.MolFromSmiles(s)
    mol = standardize(mol)

    if mol:

        mol, confIDs, energies = genConf(mol)

        return (mol, confIDs)

    else:

        return None


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

        confIDs = AllChem.EmbedMultipleConfs(mol, 50)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        # mol = rdmolops.RemoveHs(mol)

        # Chem.rdMolAlign.AlignMolConformers(mol)

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

    ############

    dists = np.array([np.linalg.norm(new_bead_location - p) for p in unaccounted_atom_pos])

    scores = 1 / (1 + np.exp(-2 * (dists - dist / 2)))

    bead_dists = np.array([np.linalg.norm(new_bead_location - b) for b in beads])

    repulsion = 100000 / (1 + np.exp(4 * (min(bead_dists) - R / 4)))

    return np.sum(scores) + repulsion


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


def make_representation(args):

    total_beads , m, bead_dist, confids = args

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
            charges.append(abs(float(a.GetProp("_GasteigerCharge"))))

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


def make_representation_morfeus(total_beads, m, bead_dist,confids):

    reps = []

    for i, beads in zip(confids, total_beads):

        atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m,[i])

        #####

        is_donor_atom, is_acceptor_atom = match_to_substructures(m)

        ############# next job is to "color the beads in"

        # define some atomwise properties

        ComputeGasteigerCharges(m)

        charges = []

        for a in m.GetAtoms():
            charges.append(float(a.GetProp("_GasteigerCharge")))

        xtb = XTB(atomic_nums, atom_coords)

        from matplotlib import pyplot as plt

        c_ = xtb.get_charges()

        c = np.array([ c_[k] for k in sorted(c_.keys())])

        charges = (c + np.array(charges))/2

        dispersion = Dispersion(atomic_nums, atom_coords)

        atom_p_int_ = dispersion.atom_p_int

        atom_p_int= np.array([ atom_p_int_[k] for k in sorted(atom_p_int_.keys())])

        atom_areas_ = dispersion.atom_areas

        atom_areas= np.array([ atom_areas_[k] for k in sorted(atom_areas_.keys())])

        nucleophilicity_ = xtb.get_fukui('nucleophilicity')

        nucleophilicity = np.array([ nucleophilicity_[k] for k in sorted(nucleophilicity_.keys())])

        electrophilicity_ = xtb.get_fukui('electrophilicity')

        electrophilicity = np.array([ electrophilicity_[k] for k in sorted(electrophilicity_.keys())])

        sasa = SASA(atomic_nums, atom_coords)

        atom_sa = sasa.atom_areas

        atom_sa = np.array([ atom_sa[k] for k in sorted(atom_sa.keys())])

        atom_vo = sasa.atom_volumes

        atom_vo = np.array([ atom_vo[k] for k in sorted(atom_vo.keys())])

        electo_avalibility =  electrophilicity * atom_sa

        nucleo_avalibility = nucleophilicity * atom_sa

        #atom_sa = sasa.atom_volumes

        charges = np.array(charges)

        CC = np.array(rdMolDescriptors._CalcCrippenContribs(m))

        #ASA = np.array([k for k in rdMolDescriptors._CalcLabuteASAContribs(m)[0]])

        #TPSA = np.array(rdMolDescriptors._CalcTPSAContribs(m))

        logP_c = CC[:, 0]

        MR_c = CC[:, 1]

        ###### next make the representation

        representation = np.zeros((len(beads), 15))

        # find distances to atoms from beads

        bead_dists_to_atoms = np.array([np.linalg.norm(b - atom_coords, axis=1) for b in beads])

        # find normalised vectors to atoms from beads

        # vectors_to_atoms = np.array([b - atom_coords for b in beads])
        # vectors_to_atoms /= np.linalg.norm(vectors_to_atoms, axis=2)[:, :, None]

        ind1 = 0

        for b, ds in zip(beads, bead_dists_to_atoms):

            weights = 1 / (1 + np.exp(2 * (ds - bead_dist / 2)))

            counts = ds < bead_dist/2

            if np.sum(counts) == 0:

                counts = ds == np.min(ds)

            # make a vector of charges

            charge_vector = np.var(weights * charges)

            representation[ind1, 0] = charge_vector

            # could have a dipole vector too
            # dipole_vectors = np.sum(weights * charges * dist,axis = 1)
            # mass vectors

            mass_vectors = np.sum(weights * atom_masses)

            representation[ind1, 1] = mass_vectors

            # logP vectors

            logP_vectors = np.average(  logP_c , weights = weights)

            representation[ind1, 2] = logP_vectors

            # MR vectors - steric descriptors

            MR_vector = np.average(  MR_c,weights = weights)

            representation[ind1, 3] = MR_vector

            # morfeus Surface area - surface area descriptors

            #ASA_vector = np.sum(weights * ASA)

            #representation[ind1, 4] = ASA_vector

            representation[ind1,4] = np.sum(atom_sa * weights)

            # morfeus volume - surface volume descriptors

            #TPSA_vector = np.sum(weights * TPSA)

            #representation[ind1, 5] = TPSA_vector

            representation[ind1,5] = np.sum(atom_vo * weights)

            # aromatic

            Aromatic_vector = np.sum(weights * atom_aromatic)

            representation[ind1, 6] = Aromatic_vector

            # HDB

            HBD_vector = np.sum(weights * is_donor_atom)

            representation[ind1, 7] = HBD_vector

            # HBA

            HDA_vector = np.sum(weights * is_acceptor_atom)

            representation[ind1, 8] = HDA_vector


            #morfeus dispersion descriptors

            representation[ind1, 9] = np.sum(weights* atom_p_int)

            representation[ind1, 10] = np.sum(weights*atom_areas)

            #

            representation[ind1,11] = np.min(counts * nucleophilicity)

            representation[ind1,12] = np.min(counts * electrophilicity)


            #

            representation[ind1,13] = np.min(counts * nucleo_avalibility)

            representation[ind1,14] = np.min(counts * electo_avalibility)

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

        w_ = atom_masses > 2

        atom_coords = atom_coords[w_]

        atomic_nums = atomic_nums[w_]

        all_aromatic = all_aromatic[w_]

        atom_masses = atom_masses[w_]

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
            write_mol_xyz(m, beads, "mol_" + str(counter), 0)

        t_beads.append(beads)

    return t_beads


def write_xyz(rep1, rep2, round_):
    f = open("tes_folder/round" + str(round_) + "_rep" + str(uuid.uuid4()) + ".xyz", "w")

    f.write(str(len(rep1) + len(rep2)) + "\n" + "\n")

    for coords in rep1:
        f.write(str("C") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in rep2:
        f.write(str("N") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def write_xyz_3(rep1, rep2, rep3, round_):
    f = open("tes_folder/helper" + str(round_) + "_rep" + str(uuid.uuid4()) + ".xyz", "w")

    f.write(str(len(rep1) + len(rep2) + len(rep3)) + "\n" + "\n")

    for coords in rep1:
        f.write(str("C") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in rep2:
        f.write(str("N") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in rep3:
        f.write(str("Cl") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

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

    D = pairwise_distances(coords_1, coords_2)

    weights = 1 / (1 + np.exp((D - R / 2)))

    '''
    residual = np.sum(abs(rep1)) + np.sum(abs(rep2))

    for i, b1 in enumerate(rep1):

        for j, b2 in enumerate(rep2):

            if i <= j:

                residual -= np.sum(weights[i, j] * (abs(b1 + b2) - 0.5 * abs(b1 - b2)))

    '''

    res_sum = np.sum(np.abs(rep1)) + np.sum(np.abs(rep2))

    temp = np.abs(rep1[:, np.newaxis, :] + rep2) - 0.5 * np.abs(rep1[:, np.newaxis, :] - rep2)

    residual = (res_sum - np.sum(weights[:, :, None] * temp)) / res_sum

    return residual


def allign_reps(beads_1, beads_2, rep1, rep2):
    # align centres of masses of beads first

    av_b1 = np.mean(beads_1, axis=0)
    av_b2 = np.mean(beads_2, axis=0)

    CoM = av_b2 - av_b1

    fit_params = Parameters()

    fit_params.add("p0", value=0, min=0, max=2 * np.pi, vary=True)
    fit_params.add("p1", value=0, min=0, max=2 * np.pi, vary=True)

    fit_params.add("x", value=CoM[0], vary=True)
    fit_params.add("y", value=CoM[1], vary=True)
    fit_params.add("z", value=CoM[2], vary=True)

    out = minimize(align_residual, fit_params,
                   args=(beads_1, beads_2, rep1, rep2),
                   method='nelder')

    #initial_res = align_residual(fit_params, beads_1, beads_2, rep1, rep2)

    aligned_beads_2 = transform(beads_2, out.params['p0'], out.params['p1'], out.params['x'], out.params['y'],
                                out.params['z'])

    final_res = align_residual(fit_params, beads_1, aligned_beads_2, rep1, rep2)

    #final_res = ( initial_res - final_res ) / initial_res

    return aligned_beads_2, final_res, out.params


def train_RF(train_descs, test_descs, train_IC50, test_IC50):
    train_fps = np.array([i for i in train_descs])

    test_fps = np.array(test_descs)

    train_y = np.array(train_IC50)

    test_y = np.array(test_IC50)

    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)[0]

    return test_pred, test_y


def AlignWorker(args):

    pairs = args[0]
    conf_beads_ = args[1]
    conf_reps_ = args[2]

    res = np.zeros(len(pairs))

    p = 0

    for pair in tqdm.tqdm(pairs):

        i = pair[0]
        j = pair[1]

        if i != j:
            i_conf_beads, j_conf_beads = conf_beads_[i], conf_beads_[j]
            i_conf_rep, j_conf_rep = conf_reps_[i], conf_reps_[j]

            temp_align_beads, residual, out_params = allign_reps(i_conf_beads, j_conf_beads, i_conf_rep, j_conf_rep)

            res[p] = residual

        p+=1

    print(os.getpid(), "finished")

    return res


if __name__ == '__main__':

    os.system("export MKL_NUM_THREADS=1")
    os.system("export OMP_NUM_THREADS=1")

    maxproc = 5

    property = 'b_ratio'

    df = pd.read_csv("/Users/alexanderhowarth/Desktop/total_b_ratio.csv").dropna(subset=property)

    df = df.drop_duplicates(subset='Compound_ID')

    code = "TRB0005601 series"

    df = df[df["STRUCTURE_COMMENT"] == code]

    IC50 = []
    smiles = []

    for i, r in tqdm.tqdm([(i, r) for i, r in df.iterrows()]):

        IC50.append(r[property])

        smiles.append(r['Smiles'])

    mols = []

    confs = []

    n_conformers = 0

    print("embedding mol")

    for s in tqdm.tqdm(smiles):

        mol, confIDs = embed_mol_smiles(s)

        mols.append(mol)

        confs.append([i for i in confIDs])

        n_conformers += len(confs[-1])

    for i, m in enumerate(mols):

        all_masses, all_coords, all_atom_number, all_aromatic = make_mol(m, confs[i])

        direction_vector, origin = find_basis(all_coords, all_masses)

        change_basis(m, confs[i], direction_vector, origin)

    # next make beads and all representations for all conformers and mols

    total_beads = []

    total_reps = []

    all_reps = []

    all_mol_inds = []

    all_beads = []

    conf_reps = []

    conf_beads = []

    print("making reps")

    for i, m in tqdm.tqdm(enumerate(mols)):

        beads = make_beads(m, confs[i], R, i)

        total_beads.append(beads)

    args = []

    for i , m  in enumerate(mols):

        args.append((total_beads[i] , m , R , confs[i]))

    ######


    to_do = ['' for i in args]

    res = ['' for i in args]

    pool = mp.Pool(maxproc)

    for i, a in enumerate(args):
        to_do[i] = pool.apply_async(make_representation_morfeus, a)

    start_time = time.time()

    for i in range(len(args)):
        res[i] = to_do[i].get()

    pool.close()

    pool.join()

    i = 0

    for rep ,beads  in zip(res,total_beads):

        total_reps.append(rep)

        for r in rep:

            conf_reps.append(r)

            all_reps.extend([r_ for r_ in r])

            all_mol_inds.append(i)

        for b in beads:

            conf_beads.append(b)

            all_beads.extend([b_ for b_ in b])

        i+=1

    # normalise the reps

    scaler = StandardScaler()

    scaler.fit(all_reps)

    all_reps = scaler.transform(all_reps)

    for i, mol_reps in enumerate(total_reps):

        for j, mol_r in enumerate(mol_reps):

            total_reps[i][j] = scaler.transform(mol_r )

    ###### find reference by all on all alignment

    '''
    chunk_number = 8

    pairs_of_confs = np.array(list(itertools.product(np.arange(n_conformers), np.arange(n_conformers))))

    chunks = np.array_split(pairs_of_confs, chunk_number)

    args = [[(c, conf_beads, conf_reps)] for c in chunks]

    p = mp.Pool()

    # defaults to os.cpu_count() workers

    maxproc = 8

    to_do = ['' for i in args]

    res = ['' for i in args]

    pool = mp.Pool(maxproc)

    for i, a in enumerate(args):
        to_do[i] = pool.apply_async(AlignWorker, a)


    start_time = time.time()

    for i in range(len(args)):
        res[i] = to_do[i].get()

    pool.close()

    pool.join()

    stop_time = time.time()

    print(stop_time - start_time, "n cpus = 8", "n alignments = ", n_conformers*n_conformers)

    # build Alignment matrix

    residuals = np.zeros((n_conformers, n_conformers))

    for c, r in zip(chunks, res):

        for pair, r_ in zip(c, r):
            residuals[pair[0], pair[1]] = r_
            residuals[pair[1], pair[0]] = r_

    pickle.dump(residuals, open("EML4_residuals.p", "wb"))
    residuals = pickle.load(open("EML4_residuals.p", "rb"))

    print(residuals)

    current_ref = np.argmin(np.sum(residuals, axis=1))

    print("ref", current_ref)

    res_dev = np.std(residuals[current_ref, :])

    print("res stddev", res_dev)

    total_aligned_beads = []
    total_aligned_reps = []

    all_alligned_beads = []

    all_mol_inds = np.array(all_mol_inds)

    i = 0

    for i_beads, i_rep in tqdm.tqdm(zip(total_beads, total_reps), total=len(total_beads)):
        ####find the confermer of mol i that has the lowest residual to the reference
        #### if it is low enough align it

        w_mol = np.where(all_mol_inds == i)[0]

        arg_min_res = np.argmin(residuals[current_ref, w_mol])
        min_res = residuals[current_ref, arg_min_res]

        #### next check if this is lower than the stdev if true do this alignment

        temp_align_beads, residual, out_params = allign_reps(conf_beads[current_ref],
                                                             conf_beads[w_mol[arg_min_res]], conf_reps[current_ref],
                                                             conf_reps[w_mol[arg_min_res]])

        write_xyz(temp_align_beads, conf_beads[current_ref], 2)

        total_aligned_beads.append(temp_align_beads)

        total_aligned_reps.append(conf_reps[w_mol[arg_min_res]])

        i += 1

    ######'''

    ##### find reference utiling tvserky similarity

    ###### find best reference

    # start by finding mol with highest similarity to all the others

    # use tversky similarity so that bits in the reference but not in the probs are not counted

    sims = np.zeros((len(mols), len(mols)))

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol_i, 2, 2048) for mol_i in mols]

    for j, mol_j in enumerate(mols):
        mol_j_fp = AllChem.GetMorganFingerprintAsBitVect(mol_j, 2, 2048)

        s = DataStructs.BulkTverskySimilarity(mol_j_fp, fps, a=1, b=0)

        sims[j, :] = s

    simsum = np.sum(sims, axis=0)

    max_sim = np.argmax(simsum)

    chunk_number = 8

    conf_inds = np.arange(n_conformers)

    conf_inds = conf_inds[all_mol_inds != max_sim]

    print("found ref", max_sim, )

    pairs_of_confs = np.array(list(itertools.product(confs[max_sim], conf_inds)))

    chunks = np.array_split(pairs_of_confs, chunk_number)

    args = [[(c, conf_beads, conf_reps)] for c in chunks]

    p = mp.Pool()

    # defaults to os.cpu_count() workers


    print("doing alignment")

    to_do = ['' for i in args]

    res = ['' for i in args]

    pool = mp.Pool(maxproc)

    for i, a in enumerate(args):
        to_do[i] = pool.apply_async(AlignWorker, a)

    start_time = time.time()

    for i in range(len(args)):
        res[i] = to_do[i].get()

    pool.close()

    pool.join()

    stop_time = time.time()

    print(stop_time - start_time, "n cpus = " ,maxproc, "n alignments = ", len(confs[max_sim]) * n_conformers)

    # build Alignment matrix

    residuals = np.zeros((len(confs[max_sim]), n_conformers))

    for c, r in zip(chunks, res):

        for pair, r_ in zip(c, r):

            residuals[pair[0], pair[1]] = r_

    # build Alignment matrix

    residuals_ = residuals[ :,all_mol_inds != max_sim ]
    current_ref = np.argmin(np.sum(residuals_, axis=1))

    current_ref = np.where(all_mol_inds == max_sim)[0][current_ref]

    # res_dev = np.std(residuals[current_ref, :])

    # print("res stddev", res_dev)

    total_aligned_beads = []
    total_aligned_reps = []

    all_alligned_beads = []

    all_mol_inds = np.array(all_mol_inds)

    i = 0

    for i_beads, i_rep in tqdm.tqdm(zip(total_beads, total_reps), total=len(total_beads)):
        ####find the confermer of mol i that has the lowest residual to the reference
        #### if it is low enough align it

        w_mol = np.where(all_mol_inds == i)[0]

        arg_min_res = np.argmin(residuals[current_ref, w_mol])

        min_res = residuals[current_ref, arg_min_res]

        #### next check if this is lower than the stdev if true do this alignment

        temp_align_beads, residual, out_params = allign_reps(conf_beads[current_ref],
                                                             conf_beads[w_mol[arg_min_res]], conf_reps[current_ref],
                                                             conf_reps[w_mol[arg_min_res]])

        total_aligned_beads.append(temp_align_beads)

        total_aligned_reps.append(conf_reps[w_mol[arg_min_res]])

        i += 1

    ####draw a picture of these beads

    total = 0

    max_beads = 0

    for beads in total_aligned_beads:

        total += len(beads)

        if len(beads) > max_beads:
            max_beads += (len(beads) - max_beads)

    all_aligned_beads = []

    all_aligned_beads_mol_index = []

    m_count = 0

    for beads in total_aligned_beads:

        for b in beads:
            all_aligned_beads.append(b)

            all_aligned_beads_mol_index.append(m_count)

        m_count += 1

    distance_matrix = np.zeros((len(all_aligned_beads), len(all_aligned_beads)))

    for i, b_i in enumerate(all_aligned_beads):

        i_mol = all_aligned_beads_mol_index[i]

        for j, b_j in enumerate(all_aligned_beads):

            j_mol = all_aligned_beads_mol_index[j]

            if i == j:

                distance_matrix[i, j] = 0
                distance_matrix[j, i] = 0

            elif i_mol == j_mol:

                distance_matrix[i, j] = 1000000000000

                distance_matrix[j, i] = 1000000000000

            elif i < j:

                d = np.linalg.norm(b_i - b_j)

                distance_matrix[i, j] = d

                distance_matrix[j, i] = d

    '''
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
    '''

    from sklearn.cluster import AgglomerativeClustering

    AggCluster = AgglomerativeClustering(distance_threshold=R / 2, n_clusters=None, affinity='precomputed',
                                         linkage='average'
                                         )

    clusters = AggCluster.fit(distance_matrix)

    cluster_ids = clusters.labels_


    n_clusters = len(set(cluster_ids))

    print("n clusters", n_clusters)

    c_count = 0

    for beads in total_aligned_beads:

        c_ids = []

        for b in beads:
            c_ids.append(cluster_ids[c_count])

            c_count += 1

    for b, id in zip(all_aligned_beads, cluster_ids):
        plt.plot(b[0], b[1], "o", color=colors[id % 10], alpha=0.5)

    plt.show()

    ###next sort all the representations the same way

    sorted_reps = []

    count = 0

    for rep in total_aligned_reps:

        sorted_reps.append(np.zeros((n_clusters, 15)))

        for r in rep:
            sorted_reps[-1][cluster_ids[count]] = r

            count += 1

    sorted_reps = np.array(sorted_reps)

    sorted_reps = [s.flatten() for s in sorted_reps]

    Rs = []

    av_vs = []

    av_ps = []

    std_ps = []

    for i in range(len(mols)):

        train_IC50 = IC50[:i] + IC50[min(i + 1, len(IC50)):]

        test_IC50 = IC50[i]

        train_descs = sorted_reps[:i] + sorted_reps[min(i + 1, len(mols)):]

        test_descs = [sorted_reps[i]]

        test_vs = []

        test_ps = []

        for j in range(0, 2):
            # make predictions

            test_pred, test_val = train_RF(train_descs, test_descs, train_IC50, test_IC50)

            test_ps.append(test_pred)

            test_vs.append(test_val)

        std_ps.append(np.std(np.abs(np.array(test_vs) - np.array(test_ps))))
        av_vs.append(test_vs[0])
        av_ps.append(np.mean(test_ps, axis=0))

    r2 = r2_score(av_vs, av_ps)

    reg = LinearRegression().fit(np.array([[a] for a in av_vs]), av_ps)

    stddev_v = np.std(av_vs)

    rmse = mean_squared_error(av_vs, av_ps)

    rmse_med = mean_squared_error(av_vs, [np.median(av_vs) for j in av_vs])

    std_errors = np.std([abs(v - p) for v, p in zip(av_vs, av_ps)])

    plt.title("Beads on a string RF Model " + code + " n = " + str(len(IC50)))
    plt.plot(av_vs, av_ps, "o",
             label="R2 = " + str(round(r2, 2)) + "\nstd = " + str(round(std_errors, 2)) + "\nRMSE = " + str(
                 round(rmse, 2)) + "\nRMSE/stddev(values) = " + str(
                 round(rmse / stddev_v, 2)) + "\nRMSE/RMSE(no skill model) = " + str(round(rmse / rmse_med, 2)),
             alpha=0.8)
    plt.plot([min(av_vs), max(av_vs)], [min(av_vs), max(av_vs)], linestyle=":", color='grey')

    plt.plot([min(av_vs), max(av_vs)],
             [reg.coef_ * min(av_vs) + reg.intercept_, reg.coef_ * max(av_vs) + reg.intercept_],
             color="C0")

    plt.legend(

    )

    plt.xlabel("Experimental")
    plt.ylabel("Predicted")

    # plt.savefig(folder + "/" + r['ID'] + ".png")

    plt.show()
    plt.close()
