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
import itertools


# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")
# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

###### have to run export MKL_NUM_THREADS=1 and export OMP_NUM_THREADS=1 before running

basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

R = 2 * 2.1943998623787615

os.system("export MKL_NUM_THREADS=1")
os.system("export OMP_NUM_THREADS=1")


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

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

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


def embed_beads_energies(mol):

    ref_mol1, ref_confIDs1 = embed_mol(mol)

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

        # atom_masses \= np.sum(atom_masses)

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


def find_basis(coordinates):
    origin = np.average(coordinates, axis=0)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return direction_vector


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

    weight_sum = np.sum(D_weights * P_weights)

    weights = P_weights[:,:,None]*D_weights[:, :, None] / weight_sum

    '''
    if approx == True:

        residual = (res_sum -   np.sum( P_weights[:,:,None]*D_weights[:, :, None] * rep_overlap ))/(res_sum  )
    '''

    if approx == True:

        residual = (res_sum - np.sum(weights * rep_overlap)) / (res_sum)

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


def Physchem_calc(m):
    return np.array([Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumHBD(m),
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

# make the reference molecule and representation

scaler = pickle.load(open(os.path.expanduser("BOAW_Mcule_morfeus_scaler_laptop.p"),"rb"))

#database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"

#scaler = pickle.load(open(os.path.expanduser("BOAW_Mcule_morfeus_scaler.p"),"rb"))

ref_mol = Chem.MolFromSmiles("CCS(C(N1C(/C2=C/c(cc3)cc(OC)c3OC(c3ccccc3)=O)=N)=NSC1=NC2=O)(=O)=O")

ref_mol, ref_confIDs, ref_beads, ref_pops = embed_beads_energies(ref_mol)

# make rep

ref_reps = []

for r_beads, r_confid in zip(ref_beads, ref_confIDs):

    rep = make_representation_morfeus(r_beads, ref_mol, R, r_confid)

    rep = scaler.transform(rep)

    ref_reps.append(rep)

ref_flat_rep, ref_flat_beads, ref_flat_pop = flatten_desc(ref_reps, ref_beads, ref_pops)

##########

# physchem filters

# database_sdf

rescore = []

for m_ in range(10):

    #prob_mol = Chem.MolFromSmiles("[O-][N+](c(cc1)ccc1C(NN(C(c(cc1)c(c2ccc3[N+]([O-])=O)c3c1[N+]([O-])=O)=O)C2=O)=O)=O")

    prob_mol = Chem.MolFromSmiles("CCS(C(N1C(/C2=C/c(cc3)cc(OC)c3OC(c3ccccc3)=O)=N)=NSC1=NC2=O)(=O)=O")

    prob_mol = standardize(prob_mol)

    if prob_mol:

        prob_mol, prob_confIDs, prob_beads, prob_pops = embed_beads_energies(prob_mol)

        #make rep

        prob_reps =[ ]

        for p_beads, p_confid in zip(prob_beads, prob_confIDs):

            rep = make_representation_morfeus(p_beads, prob_mol, R, p_confid)

            rep = scaler.transform(rep)

            prob_reps.append(rep)

        prob_flat_rep, prob_flat_beads,prob_flat_pops  = flatten_desc(prob_reps, prob_beads, prob_pops)

        #align beads to each set of ref_beads

        aligned_beads_2, final_res, params = allign_reps(ref_flat_beads, prob_flat_beads, ref_flat_rep, prob_flat_rep,
                                                         ref_flat_pop, prob_flat_pops,approx=True)

        rescore.append(final_res)

        print(final_res)