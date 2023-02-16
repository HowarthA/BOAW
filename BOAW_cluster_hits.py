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


def embed_mol(mol):

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

        confIDs = AllChem.EmbedMultipleConfs(mol, 2)

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

        atom_inds = np.arange(len(atom_coords))

        for b, ds in zip(beads, bead_dists_to_atoms):

            weights = 1 / (1 + np.exp(2 * (ds - bead_dist / 2)))

            counts = ds < bead_dist/2

            if np.sum(counts) == 0:

                counts = ds ==np.min(ds)

            # weights = 1 / (1 + np.exp(  (ds -  bead_dist)))
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

            # ASA - surface area descriptors

            #ASA_vector = np.sum(weights * ASA)

            #representation[ind1, 4] = ASA_vector

            representation[ind1,4] = np.sum(atom_sa * weights)

            # TPSA - surface area descriptors

            #TPSA_vector = np.sum(weights * TPSA)

            #representation[ind1, 5] = TPSA_vector

            representation[ind1,5] = np.sum(atom_vo * weights)

            # aromatic

            Aromatic_vector = np.sum(weights * atom_aromatic)

            representation[ind1, 6] = Aromatic_vector

            # HDB

            HDB_vector = np.sum(weights * is_donor_atom)

            representation[ind1, 7] = HDB_vector

            # HBA

            HDA_vector = np.sum(weights * is_acceptor_atom)

            representation[ind1, 8] = HDA_vector

            representation[ind1, 9] = np.sum(weights* atom_p_int)

            representation[ind1, 10] = np.sum(weights*atom_areas)

            representation[ind1,11] = np.min(counts * nucleophilicity)

            representation[ind1,12] = np.min(counts * electrophilicity)

            representation[ind1,13] = np.max(counts * nucleo_avalibility)

            representation[ind1,14] = np.max(counts * electo_avalibility)

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

    # initial_res = align_residual(fit_params, beads_1, beads_2, rep1, rep2)

    aligned_beads_2 = transform(beads_2, out.params['p0'], out.params['p1'], out.params['x'], out.params['y'],
                                out.params['z'])

    final_res = align_residual(fit_params, beads_1, aligned_beads_2, rep1, rep2)

    # final_res = ( initial_res - final_res ) / initial_res

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

    pair = args[0]

    conf_beads_1 = args[1][pair[0]]
    conf_beads_2 = args[1][pair[1]]

    conf_reps_1 = args[2][pair[0]]
    conf_reps_2 = args[2][pair[1]]

    res_ = np.zeros((len( conf_beads_1 ) , len(conf_beads_2)))

    i = 0

    for  rep1i , beads1i in zip(conf_reps_1,conf_beads_1):

        j=0

        for rep2j , beads2j in zip(conf_reps_2,conf_beads_2):

            temp_align_beads, residual, out_params = allign_reps(beads1i, beads2j, rep1i, rep2j)

            res_[ i,j] = residual

            j+=1

        i+=1

    return np.min(res_)


if __name__ == '__main__':

    ### input mols as a list of smiles or sdf files


    mols_ = [ standardize(m) for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/YAP.sdf")]


    mols = []

    confs = []

    n_conformers = 0

    for m in tqdm.tqdm(mols_):

        mol, confIDs = embed_mol(m)

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

    for i, m in tqdm.tqdm(enumerate(mols)):

        beads = make_beads(m, confs[i], R, i)

        rep = make_representation_morfeus(beads, m, R, confs[i])

        total_beads.append(beads)
        total_reps.append(rep)

        for r in rep:

            conf_reps.append(r)

            all_reps.extend([r_ for r_ in r])

            all_mol_inds.append(i)

        for b in beads:
            conf_beads.append(b)

            all_beads.extend([b_ for b_ in b])

    # normalise the reps

    scaler = StandardScaler()

    scaler.fit(all_reps)

    all_reps = scaler.transform(all_reps)

    for i, mol_reps in enumerate(total_reps):

        for j, mol_r in enumerate(mol_reps):
            
            total_reps[i][j] = scaler.transform(mol_r)

    ###### find reference by all on all alignment

    chunk_number = 8

    pairs_of_mols = np.array(list(itertools.product(np.arange(len(mols)), np.arange(len(mols)))))

    args = [[(c, total_beads, total_reps)] for c in pairs_of_mols]

    print(len(args))

    p = mp.Pool()

    # defaults to os.cpu_count() workers

    maxproc = 5

    to_do = ['' for i in args]

    print("running alignments", print(len(to_do)))

    res_vector = ['' for i in args]

    pool = mp.Pool(maxproc)

    for i, a in enumerate(args):
        to_do[i] = pool.apply_async(AlignWorker, a)

    start_time = time.time()

    for i in range(len(args)):
        res_vector[i] = to_do[i].get()

    pool.close()

    pool.join()

    stop_time = time.time()

    print(stop_time - start_time, "n cpus = ",maxproc, "n alignments = ", n_conformers*n_conformers)

    # build Alignment matrix

    distance_matrix = np.zeros((len(mols), len(mols)))

    for c, r in zip(pairs_of_mols, res_vector):

        print(c,r)

        distance_matrix[c[0], c[1]] = r
        distance_matrix[c[1], c[0]] = r

    pickle.dump(distance_matrix, open("residuals.p", "wb"))
    distance_matrix = pickle.load(open("residuals.p", "rb"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    corr = np.exp(- distance_matrix)

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()

