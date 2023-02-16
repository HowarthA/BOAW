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


# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")
# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

###### have to run export MKL_NUM_THREADS=1 and export OMP_NUM_THREADS=1 before running

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


def embed_mol(s):

    mol = Chem.MolFromSmiles(s)

    mol = standardize(mol)

    mol = rdmolops.AddHs(mol)

    if mol:

        confIDs = AllChem.EmbedMultipleConfs(mol, 10)

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

        return (mol, confIDs)

    else:

        return None


def embed_mol_smiles_single_conf(s):

    mol = Chem.MolFromSmiles(s)
    mol = standardize(mol)

    mol = rdmolops.AddHs(mol)

    if mol:

        AllChem.EmbedMolecule(mol)

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)

        except:
            return None

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

        return mol
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


def find_basis(coordinates, masses):
    origin = np.average(coordinates, axis=0, weights=masses)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return direction_vector, origin


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

        representation[ind1,13] = np.max(counts * nucleo_avalibility)

        representation[ind1,14] = np.max(counts * electo_avalibility)

        ind1 += 1

    return np.array(representation)


def make_beads(m, confIDs, dist):
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    t_beads = []

    #### rotate mol onto PC axis

    for i in confIDs:

        atom_masses, atom_coords, atomic_nums, all_aromatic = make_mol(m, i)

        # atom_masses \= np.sum(atom_masses)

        w_ = atom_masses>2

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

    residual = (res_sum -  np.sum(weights[:, :, None] * temp))/res_sum

    return residual


def allign_reps(beads_1, beads_2, rep1, rep2):

    #align centres of masses of beads first

    av_b1 = np.mean(beads_1,axis = 0)
    av_b2 = np.mean(beads_2,axis = 0)

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

    return aligned_beads_2, final_res, out.params


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


def rep_sim(smiles1):

    mol1, confIDs = embed_mol(smiles1  )

    if mol1:

        beads =  make_beads(mol1, confIDs, R)

        reps = []

        for i in confIDs:

            rep1 = make_representation_morfeus(beads[i], mol1, R, i)
            rep1 = scaler.transform(rep1)

            reps.append(rep1)

        res_c = np.ones((50,50))

        #then align all pairs:

        for c1 in confIDs:

            for c2 in confIDs[c1:]:

                aligned_beads, sim, params = allign_reps(beads[c1], beads[c2], reps[c1], reps[c2])

                res_c[c1,c2] = sim
                res_c[c2,c1] = sim

    return np.mean(res_c) , np.std(res_c)



# make the reference molecule and representation

#database = os.path.expanduser("~/mcule_purchasable_in_stock_221205.smi")
#scaler = pickle.load(open(os.path.expanduser("~/BOAW_Mcule_morfeus_scaler.p"),"rb"))

database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"
scaler = pickle.load(open(os.path.expanduser("BOAW_Mcule_morfeus_scaler.p"),"rb"))

database = open(database, "r").readlines()

db_length = len(database)

os.system("export MKL_NUM_THREADS=1")
os.system("export OMP_NUM_THREADS=1")

average_std = 0

old_average_std = 1



old_average_std = 1

average_mean= 0


loop = 0



sample_n = 100

measured_stds = []

measured_means = [ ]


while round(average_std,4) != round(old_average_std,4):

    print("loop", loop, "std" ,average_std ,"mean",average_mean, sample_n )

    #make a sample of n pairs of molecules

    smiles_list = np.random.randint(0,db_length-1,sample_n)

    stds = np.ones(sample_n)

    means = np.ones(sample_n)

    p = 0

    for s1 in tqdm.tqdm(smiles_list):

        sm1 = database[s1].split()[0]

        std,mean = rep_sim(sm1)

        stds[p] = std

        means[p] = mean

        p+=1


    measured_stds.extend(stds[ stds != np.nan ] )
    measured_means.extend(means[stds != np.nan ])

    old_average_std = copy.copy(average_std)

    average_std = np.std(measured_stds )

    average_mean = np.mean(measured_means)

    loop+=1

