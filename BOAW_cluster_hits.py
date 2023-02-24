import itertools
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
import tqdm
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn import decomposition
from rdkit.Chem import rdMolDescriptors
from lmfit import Parameters, fit_report, minimize
from rdkit.Chem import rdMolAlign
from scipy.spatial.transform import Rotation
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Geometry import Point3D
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
import pickle
import copy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from morfeus import XTB
from morfeus import Dispersion
from morfeus import SASA
from matplotlib import pyplot as plt
import multiprocessing as mp
import time

# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")
# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

###### have to run export MKL_NUM_THREADS=1 and export OMP_NUM_THREADS=1 before running

os.system("export MKL_NUM_THREADS=1")
os.system("export OMP_NUM_THREADS=1")

basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

R = 2 * 2.1943998623787615

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def write_mol_xyz(mol, kmeans, title,cid):
    f = open(title  + ".xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), mol.GetConformer(cid).GetPositions()):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()

def draw_all_beads(beads,title):

    N = 0

    for bead_set in beads:

        N+=len(bead_set)

    f = open(title  + ".xyz", "w")

    f.write(str(N) + "\n" + "\n")

    for bead_set in beads:
        for b in bead_set:
            f.write(str("O") + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + "\n")

def draw_aligned_all_beads(beads1,beads2,title):

    N = 0

    for bead_set in beads1:

        N+=len(bead_set)

    for bead_set in beads2:

        N+=len(bead_set)

    f = open(title  + ".xyz", "w")

    f.write(str(N) + "\n" + "\n")

    for b in beads1:

        f.write(str("O") + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + "\n")


    for b in beads2:

        f.write(str("N") + " " + str(b[0]) + " " + str(b[1]) + " " + str(b[2]) + "\n")


def embed_mol(mol):

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

        confIDs = AllChem.EmbedMultipleConfs(mol,n_conformers)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None


def embed_mol_n(mol,n):

    if mol:

        mol = Chem.AddHs(mol)

        # Generate conformers

        AllChem.EmbedMolecule(mol)

        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)

        return mol

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

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None



def embed_mol_Inchi(s):

    mol = Chem.MolFromInchi(s)
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

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None



def make_mol(mol, id):

    all_coords = []
    all_masses = []
    all_aromatic = []
    all_atom_number = []

    positions = mol.GetConformer(id).GetPositions()

    for atom, p in zip(mol.GetAtoms(), positions):
        all_masses.append(atom.GetMass())
        all_atom_number.append(atom.GetAtomicNum())
        all_aromatic.append(atom.GetIsAromatic())
        all_coords.append(p)

    return np.array(all_masses), np.array(all_coords), np.array(all_atom_number), np.array(all_aromatic)


def find_basis(coordinates):
    origin = np.average(coordinates, axis=0)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return direction_vector


def change_basis(m,  direction_vector, origin,c):

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

    repulsion = 100000 / (1 + np.exp(4*(min(bead_dists) - R/4 )))

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


def make_representation(beads, m, bead_dist, i):

    atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m, i)

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



    return np.array(representation)


def make_representation_morfeus(beads, m, bead_dist,i):

    atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m,i)

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

        representation[ind1,13] = np.min(counts * nucleo_avalibility)

        representation[ind1,14] = np.min(counts * electo_avalibility)

        ind1 += 1

    return np.array(representation)



def make_beads(m, confIDs, dist):
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    t_beads = []

    SSSR_ = Chem.GetSSSR(m)

    SSSR = []

    ring_atoms = []

    for r in SSSR_:

        SSSR.append([a for a  in r])

        ring_atoms.extend([a for a  in r])

    ring_atoms = list(set(ring_atoms))

    #### rotate mol onto PC axis

    for i in confIDs:

        atom_masses_, atom_coords_, atomic_nums_, all_aromatic_ = make_mol(m, i)

        ### fold rings into centre - duplicating atoms in multiple rings

        atom_inds = np.arange(0,len(atom_coords_))

        non_ring_atoms = list( set(atom_inds) - set(ring_atoms) )

        atom_coords = atom_coords_[non_ring_atoms]
        atom_masses = atom_masses_[non_ring_atoms]
        atomic_nums = atomic_nums_[non_ring_atoms]

        for r in SSSR:

            atom_coords = np.concatenate( (atom_coords ,[np.mean(atom_coords_[r], axis=0) for i in r]))

            atom_masses = np.concatenate((atom_masses, [atom_masses_[i] for i in r]))

            atomic_nums = np.concatenate((atomic_nums, [atomic_nums_[i] for i in r]))

        w_ = atom_masses>2

        atom_coords = atom_coords[w_]

        atomic_nums = atomic_nums[w_]

        #all_aromatic = all_aromatic[w_]

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

        t_beads.append(beads)

    return t_beads


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


def Physchem_calc(m):
    return np.array(
        [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
         rdMolDescriptors.CalcFractionCSP3(m)])


def Physchem_filter(prob_props, ref_props):
    mw_high = ref_props[0] + 50
    mw_low = ref_props[0] - 50

    logp_high = ref_props[1] + 1
    logp_low = ref_props[1] - 1

    f_sp3_high = ref_props[4] + 0.2
    f_sp3_low = ref_props[4] - 0.2

    return np.all(np.array(
        [prob_props[0] > mw_low, prob_props[0] < mw_high, prob_props[1] > logp_low, prob_props[1] < logp_high,
         prob_props[2] == ref_props[2], prob_props[3] == ref_props[3], prob_props[4] > f_sp3_low,
         prob_props[4] < f_sp3_high]))


def embed_beads_energies(m):

    ref_mol1, ref_confIDs1 = embed_mol(m)

    rdMolAlign.AlignMolConformers(ref_mol1)

    ref_confIDs1 = [ i for i in ref_confIDs1]

    ref_beads1 = make_beads(ref_mol1, ref_confIDs1, R)

    mp = AllChem.MMFFGetMoleculeProperties(ref_mol1, mmffVariant='MMFF94s')

    energies1 = []

    for cid in ref_confIDs1:

        ff = AllChem.MMFFGetMoleculeForceField(ref_mol1, mp, confId=cid)
        e = ff.CalcEnergy()
        energies1.append(e)

    gasConstant = 8.3145
    temperature = 298.15
    kcalEnergy = 4184

    relEs1 = np.array(energies1)*kcalEnergy

    populations1 = np.exp( -1* np.array(relEs1)  / (gasConstant * temperature))

    sum_pop1 = np.sum(populations1)

    populations1 = populations1/sum_pop1

    return ref_mol1, ref_confIDs1, ref_beads1, populations1







def align_residual_grid(params, coords_1, coords_2, rep1, rep2,pop1,pop2):

    # next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    x = params['x']
    y = params['y']
    z = params['z']

    coords_2 = transform(coords_2, p0, p1, x, y, z)

    ### make box

    x_ = np.arange( np.min( (np.min(coords_1[:,0]) , np.min(coords_2[:,0])) )  - R,np.max( (np.max(coords_1[:,0]) , np.max(coords_2[:,0])) )  + R    , 0.5)

    y_ = np.arange( np.min( (np.min(coords_1[:,1]) , np.min(coords_2[:,1])) )  - R,np.max( (np.max(coords_1[:,1]) , np.max(coords_2[:,1])) )  + R    , 0.5)

    z_ =np.arange( np.min( (np.min(coords_1[:,2]) , np.min(coords_2[:,2])) )  - R,np.max( (np.max(coords_1[:,2]) , np.max(coords_2[:,2])) )  + R    , 0.5)

    grid = np.vstack(np.meshgrid(x_, y_, z_)).reshape(3, -1).T

    mol1 = np.zeros( (len(x_) * len(y_) * len(z_)  , 15) )

    mol2 = np.zeros( (len(x_) * len(y_) * len(z_)  , 15) )

    for bead, r ,p  in zip(coords_1 ,rep1,pop1):

        ds = np.linalg.norm( bead - grid )

        mol1 +=  p * np.multiply.outer(  r , np.exp( - 2*ds/R))

    for bead, r , p in zip(coords_2, rep2,pop2):

        ds = np.linalg.norm(bead - grid)

        mol2 +=  p* np.multiply.outer(  r, np.exp( - 2*ds/R))

    res_sum = np.sum(np.abs(mol1)) + np.sum(np.abs(mol2))

    temp = np.sum(np.abs(mol1 + mol2) - 0.5 * np.abs(mol1 - mol2))

    residual = (res_sum -  temp)/res_sum

    return residual



def transform(rep, angles , basis , x, y, z):

    rep_ = copy.copy(rep)

    # find centre of mass of the second rep

    origin = np.average(rep_, axis=0)

    # convert bead positions into direction vectors from centre of mass

    rep_ -= origin

    # next rotate vectors to centre of mass by the fitted angles

    for i,p in enumerate(basis):

        r_ab = Rotation.from_rotvec(-1 * p * angles[i])

        rep_ = r_ab.apply(rep_)

    rep_ += origin

    # next perform translations

    rep_ += np.array([x, y, z])

    return rep_



def align_residual(params, coords_1, coords_2, rep1, rep2,pop1,pop2,basis,P_weights,res_sum,rep_overlap,approx):
    # next apply interpolated rotations

    angles = [ params['p' + str(i)] for i in range(len(basis))  ]

    x = params['x']
    y = params['y']
    z = params['z']

    coords_2 = transform(coords_2,angles, basis ,x,y,z )

    D = pairwise_distances(coords_1, coords_2)

    #approximated overlap
    #D_weights = 1 / (1 + np.exp((D - R/2 )))

    D_weights = np.exp( - 2 * D/R)

    if approx == True:

        residual = (res_sum -   np.sum( P_weights[:,:,None]*D_weights[:, :, None] * rep_overlap))/res_sum

    else:

        b = np.abs( rep1[:,np.newaxis, :]* D_weights[:,:,np.newaxis] +  rep2[np.newaxis,:,:] ) - np.abs( rep1[:,np.newaxis, :]* D_weights[:,:,np.newaxis] -  rep2[np.newaxis,:,:] )

        a = np.abs(  rep1[:,np.newaxis, :] +   rep2[np.newaxis,:,:]* D_weights[:,:,np.newaxis])  - np.abs(  rep1[:,np.newaxis, :] -  rep2[np.newaxis,:,:]* D_weights[:,:,np.newaxis])

        P_weights = np.multiply.outer(pop1,pop2)

        res_sum = np.sum(pop1[:,None]*np.abs(rep1)) + np.sum(pop2[:,None] * np.abs(rep2))

        rep_overlap =  (a + b)

        residual = (res_sum -   0.5 *np.sum( P_weights[:,:,None] * rep_overlap))/res_sum


    return residual


def allign_reps(beads_1, beads_2, rep1, rep2,pop1,pop2,approx):

    #align centres of masses of beads first

    ###

    rep1 = np.ones(np.shape(rep1))
    rep2 = np.ones(np.shape(rep2))

    av_b1 = np.mean(beads_1,axis = 0)
    av_b2 = np.mean(beads_2,axis = 0)

    CoM = av_b2 - av_b1

    fit_params = Parameters()

    #find basis

    pca1 = find_basis(beads_1)
    pca2 = find_basis(beads_2)

    #

    basis = [ np.cross(p1 , p2) / np.linalg.norm(np.cross(p1 , p2))  for p1,p2 in itertools.product(pca1,pca2) ]

    for p in range(0,len(basis)):
        fit_params.add("p" + str(p), value=0, min=-np.pi, max=np.pi, vary=True)

    ######

    fit_params.add("x", value=CoM[0], vary=True)
    fit_params.add("y", value=CoM[1], vary=True)
    fit_params.add("z", value=CoM[2], vary=True)

    P_weights = np.multiply.outer(pop1,pop2)

    res_sum = np.sum(pop1[:,None]*np.abs(rep1)) + np.sum(pop2[:,None] * np.abs(rep2))

    rep_overlap = np.abs(rep1[:, np.newaxis, :] + rep2) - np.abs(rep1[:, np.newaxis, :] - rep2)

    out = minimize(align_residual, fit_params,
                   args=(beads_1, beads_2, rep1, rep2,pop1,pop2,basis,P_weights,res_sum,rep_overlap,approx),
                   method='nelder')

    #initial_res = align_residual(fit_params, beads_1, beads_2, rep1, rep2)

    aligned_beads_2 = transform(beads_2, [ out.params['p' + str(i)] for i in range(len(basis))  ] , basis, out.params['x'], out.params['y'],
                                out.params['z'])


    return aligned_beads_2, out.residual, out.params



################ flatten the reps and beads

def flatten_desc(ref_rep1,ref_beads1,populations1):

    flat_beads1 =[]

    flat_rep1 = []

    flat_pop1= []

    l_b = []

    for rep, beads, populations in zip(ref_rep1,ref_beads1,populations1):

        for r_,b_ in zip(rep,beads):

            flat_beads1.append(b_)

            flat_rep1.append(r_)

            flat_pop1.append(populations)

    return np.array(flat_rep1),np.array(flat_beads1),np.array(flat_pop1)


def AlignWorker(args):

    c, beads, reps,pops = args

    res = []

    for c2 in range(len(reps)):

        if c2 ==c:

            res.append(0)

        else:

            beads_1 = beads[c]
            beads_2 = beads[c2]

            reps_1 = reps[c]
            reps_2 = reps[c2]

            pops_1 = pops[c]
            pops_2 = pops[c2]

            flat_rep1, flat_beads1, flat_pop1 = flatten_desc(reps_1,beads_1,pops_1)
            flat_rep2, flat_beads2, flat_pop2 = flatten_desc(reps_2,beads_2,pops_2)

            temp_align_beads, residual, out_params = allign_reps(flat_beads1, flat_beads2, flat_rep1, flat_rep2, flat_pop1,
                                                             flat_pop2,approx=False)

            res.append(residual)

    return res



if __name__ == '__main__':

    ### input mols as a list of smiles or sdf files

    labels =[]
    mols_ = []

    for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/EML4-ALK_LL1_phasscan_DR.sdf"):

        ID = m.GetProp('ID')

        if ID not in labels:

            labels.append(ID)
            mols_.append(m)


    for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/extra_EML4_hits.sdf"):

        ID = m.GetProp('ID')

        if ID not in labels:

            labels.append(ID)
            mols_.append(m)


    mols = []

    confs = []

    beads = []

    reps_ = []

    reps_all =[]

    pops = []

    n_conformers = 0

    for m in tqdm.tqdm(mols_):

        mol, confIDs, beads_, populations = embed_beads_energies(m)

        mols.append(mol)

        confs.append([i for i in confIDs])

        rep_ = []

        for b, confid in zip(beads_, confIDs):

            r = make_representation_morfeus(b, mol, R, confid)

            #r = scaler.transform(r)

            rep_.append(r)

        reps_all.extend(rep_[0])

        beads.append(beads_)

        reps_.append(rep_)

        pops.append(populations)

        n_conformers += len(confs[-1])

    scaler = StandardScaler()

    scaler.fit(reps_all)

    scaler = pickle.load(open("BOAW_LL1_morfeus_scaler.p","rb"))

    reps = []

    for rep in reps_:

        reps.append([])

        for r in rep:

            reps[-1].append(scaler.transform(r))

    # next make beads and all representations for all conformers and mols

    ###### find reference by all on all alignment

    #pairs_of_mols = np.array(list(itertools.product(np.arange(len(mols)), np.arange(len(mols)))))

    #distance_matrix = np.zeros((len(mols), len(mols)))

    #for p in tqdm.tqdm(pairs_of_mols):

     #   if p[1] != p[0]:

      #      flat_rep1, flat_beads1, flat_pop1 = flatten_desc(reps[p[0]], beads[p[0]], pops[p[0]])
       #     flat_rep2, flat_beads2, flat_pop2 = flatten_desc(reps[p[1]], beads[p[1]], pops[p[1]])

        #    aligned_beads_2, final_res, params = allign_reps(flat_beads1, flat_beads2, flat_rep1, flat_rep2, flat_pop1,
            #                                                 flat_pop2)

         #   distance_matrix[p[0], p[1]] = final_res
          #  distance_matrix[p[1], p[0]] = final_res

    maxproc = 5

    #pairs_of_mols = np.array(list(itertools.product(np.arange(len(mols)), np.arange(len(mols)))))

    args = [[(c, beads, reps,pops)] for c in range(len(mols))]

    p = mp.Pool()

    # defaults to os.cpu_count() workers

    to_do = ['' for i in args]

    res_vector = ['' for i in args]

    print("running alignments", len(mols) * len(mols))

    pool = mp.Pool(maxproc)

    for i, a in enumerate(args):
        to_do[i] = pool.apply_async(AlignWorker, a)

    start_time = time.time()

    for i in range(len(args)):
        res_vector[i] = to_do[i].get()

    pool.close()

    pool.join()

    stop_time = time.time()

    print(stop_time - start_time, "n cpus = ",maxproc, "n alignments = ", len(mols) * len(mols))

    # build Alignment matrix

    distance_matrix = np.zeros((len(mols), len(mols)))

    for c,  r in enumerate(res_vector):

        distance_matrix[c , : ] = r

    #pickle.dump(distance_matrix, open("residuals.p", "wb"))
    #distance_matrix = pickle.load(open("residuals.p", "rb"))

    pickle.dump(distance_matrix, open("EML4_residuals.p", "wb"))

    distance_matrix = pickle.load(open("EML4_residuals.p", "rb"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform



    mean_d = np.mean(distance_matrix)
    std_d = np.std(distance_matrix)

    '''
    distance_matrix[distance_matrix < 0 ] = 0


    for i in range(len(distance_matrix)):

        distance_matrix[i,i] = mean_d

    corr = np.exp(-( (distance_matrix - mean_d)/( 0.1) ) **2 )

    distance_matrix = 1-corr

    for i in range(len(distance_matrix)):

        distance_matrix[i,i] = 0
    '''

    distance_matrix[distance_matrix < 0 ] = 0
    distance_matrix = (distance_matrix + distance_matrix.T) / 2


    corr = 1-distance_matrix

    print(distance_matrix)
    # Ensure the correlation matrix is symmetric



    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.

    print(distance_matrix)

    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()
