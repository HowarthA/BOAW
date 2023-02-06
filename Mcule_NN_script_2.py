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

# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")
# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

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

        # AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)

        #Chem.rdMolAlign.AlignMolConformers(mol)

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

        confIDs = AllChem.EmbedMultipleConfs(mol, 5)

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


# make the reference molecule and representation

database = os.path.expanduser("~/mcule_purchasable_in_stock_221205.smi")

scaler = pickle.load(open(os.path.expanduser("~/BOAW_Mcule_scaler.p"),"rb"))

#database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"
#scaler = pickle.load(open(os.path.expanduser("BOAW_Mcule_scaler.p"),"rb"))

db_length = len(open(database, "r").readlines())

ref_mol, ref_confIDs = embed_mol_smiles("O=C(NS(=O)(=O)C1C[C@@H]2CC[C@H]1C2)c1cc2ccccc2o1")


# ref_mol, ref_confIDs = embed_mol_smiles("Cc1ccc(OCC(=O)Nn2c(-c3ccco3)n[nH]c2=S)c(C)c1")


##########

# physchem filters

ref_props = Physchem_calc(ref_mol)

##########

# ref_mol, ref_confIDs = embed_mol_smiles("CN(C)CCCN(C(=O)c1cccc(C(F)(F)F)c1)c1nc2ccc(F)cc2s1")

for i in ref_confIDs:

    atomic_mass, positions, atomic_numbers, atom_aromatic = make_mol(ref_mol, i)

    direction_vector, origin = find_basis(positions, atomic_mass)

    change_basis(ref_mol, direction_vector, origin, i)

# database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"

# database = "5601.smi"

ref_beads = make_beads(ref_mol, ref_confIDs, R)

ref_reps = []

for beads_, confid in zip(ref_beads, ref_confIDs):

    ref_rep_ = make_representation(beads_, ref_mol, R, confid)

    ref_rep_ = scaler.transform(ref_rep_)

    ref_reps.append(ref_rep_)

# database_sdf

maxproc = 600

threshold = 0.3

def SearchWorker(args):
    i = 0

    ref_mol = args[0]

    inds = args[1]

    proc = args[2]

    s_file = open(database, "r")

    while i < inds[0] - 1:
        s_file.readline()

        i += 1

    NNs = []

    for i in tqdm.tqdm(inds):

        s = s_file.readline()

        s = s.split()[0]

        try:

            prob_mol = Chem.MolFromSmiles(s)

            prob_mol = standardize(prob_mol)


        except:

            prob_mol = None

        if prob_mol:

            ###### calculate some physchem properties for initial filters

            prob_props = Physchem_calc(prob_mol)

            if Physchem_filter(prob_props, ref_props):

                ######

                # embed this new mol

                prob_mol = embed_mol_n(prob_mol,1 )

            else:

                prob_mol = None

            if prob_mol:

                # make beads

                prob_beads = make_beads(prob_mol, [0], R)

                #make rep

                prob_rep = make_representation(prob_beads[0], prob_mol, R, 0)

                prob_rep = scaler.transform(prob_rep)

                #align beads to each set of ref_beads

                res = []

                for ref_beads_, ref_rep_ in zip(ref_beads, ref_reps):

                    aligned_beads_2, final_res, params = allign_reps(ref_beads_, prob_beads[0], ref_rep_, prob_rep)

                    res.append(final_res)

                if np.min(res) < threshold:

                    prob_mol.SetProp("_similarity", str(np.median(np.min(res))))

                    NNs.append(prob_mol)

                    print("found ", proc, len(NNs))

    if len(NNs) > 0:

        with Chem.SDWriter("output_" + str(proc) + ".sdf") as w:

            print("writing to "  "output_" + str(proc) + ".sdf", len(NNs), proc)

            for mol_ in NNs + [ref_mol]:
                w.write(mol_)

    else:

        print("none found " + str(proc))

inds = np.arange(0, db_length)

chunks = np.array_split(inds, maxproc)

args = []

c = 0

for i, j in enumerate(chunks):
    args.append((ref_mol, j, c))

    c += 1


#SearchWorker(args[0])

import multiprocessing

p = multiprocessing.Pool()

# defaults to os.cpu_count() workers
p.map_async(SearchWorker, args)

# perform process for each i in i_list
p.close()
p.join()

# Wait for all child processes to close.

'''
from pathos import multiprocessing
pool = multiprocessing.Pool(maxproc)

for i, j in enumerate(chunks):
    args[i] = (ref_mol, j,  i)

results_ = pool.map(SearchWorker, args)
'''