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
from rdkit.Chem import Descriptors3D
import copy
from  scipy.stats import gaussian_kde
import pickle
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
import os

maxproc = 600

R = 2.1943998623787615 * 2


database = os.path.expanduser("~/mcule_purchasable_in_stock_221205.smi")

p_file = os.path.expanduser("~/mcule_sample_boundaries.p")

boundaries = pickle.load( open(p_file ,"rb"))

#database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"

###############

def bead_encode(rep):
    bead_encoding = []

    for bead in rep:

        encoding = [0 for i in range(0, 9)]

        for b_id, p in enumerate(bead):
            w_b = np.argmin(np.abs(p - boundaries[b_id]))

            encoding[b_id] = int(w_b)

        encoding_ = ''

        for b in encoding:
            encoding_ += str(b).zfill(2)

        bead_encoding.append(encoding_)

    return bead_encoding


def embed_mol_sdf(mol):

    AllChem.EmbedMolecule(mol)

    mol = rdmolops.AddHs(mol)

    converge = AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

    if converge != -1:

        return mol

    else:

        return None


def embed_mol_2d(mol):

    mol = AllChem.Compute2DCoords(mol)

    return mol

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def make_mol(molecule):

    atomic_mass = []
    atomic_numbers = []
    atom_aromatic = []
    positions = []
    symbols = []

    positions_ = molecule.GetConformer().GetPositions()

    positions.extend(positions_)

    for atom in molecule.GetAtoms():
        atomic_mass.append(atom.GetMass())
        atomic_numbers.append(atom.GetAtomicNum())
        atom_aromatic.append(atom.GetIsAromatic())
        symbols.append(atom.GetSymbol())

    return np.array(atomic_mass), np.array(positions), np.array(atomic_numbers),symbols,atom_aromatic


def write_mol_xyz(mol,coords_,kmeans):

    f = open("suramin_p_jan.xyz","w")

    N_atoms=mol.GetNumAtoms()

    f.write(str(N_atoms+ len(kmeans) ) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(),coords_):

        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for  coords in kmeans:

        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def residual2(params,atomic_nums, unaccounted_atom_pos, beads,dist,connection_bead_position):

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # get initial vectors along z principle axis

    initial_vector = basis[0]

    #next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    r0_ab = Rotation.from_rotvec(  -1 *basis[1] * p0 )
    r1_ab = Rotation.from_rotvec(  -1 *basis[2] * p1)

    initial_vector = r0_ab.apply(initial_vector)

    initial_vector = r1_ab.apply(initial_vector)

    new_bead_location = dist* initial_vector + connection_bead_position

    #next work out how many of the unnaccounted for atoms this bead now accounts for

    dists =  np.array([ np.linalg.norm(new_bead_location - p) for p in unaccounted_atom_pos ])

    scores =  1 / (1 + np.exp(-2 * (dists - dist/2))) * atomic_nums

    return np.sum(scores)


def find_basis(coordinates, masses):

    origin = np.average(coordinates, axis=0, weights=masses)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return  direction_vector , origin


def change_basis(mols , direction_vector,origin):

    for m in mols:

        coordinates = m.GetConformer().GetPositions()

        a_c = []

        for a_ in coordinates:

            a_ = np.matmul(direction_vector, a_)

            a_c.append(a_)

        atom_coords = np.array(a_c)

        atom_coords -= origin

        conf = m.GetConformer()

        for i in range(m.GetNumAtoms()):
            x, y, z = atom_coords[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))

    return mols


def make_beads(m, dist ):

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    atom_masses, atom_coords, atomic_nums,symbols, all_aromatic = make_mol(m)

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

    bead_n = 1

    bead_connections = []

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

            atoms_dist_to_beads = np.linalg.norm( unaccounted_atom_pos - new_bead_location, axis=1)

            accounted = atoms_dist_to_beads < dist

            accounted_ns.append(np.sum(accounted))

            new_beads_positions.append(new_bead_location)

        ### decide on which connection lead to the greatest increase in new atoms accounted for

        #if (len(beads) > 2) & (np.all(unnaccounted_ns == unnaccounted_ns[0])):

         #   break

        best = accounted_ns.index(max(accounted_ns))

        bead_connections.append(( best , bead_n ))

        new_bead = new_beads_positions[best]

        beads = np.vstack((beads, new_bead))

        atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in atom_coords])

        unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist

        unaccounted_atom_pos_old = len(unaccounted_atom_pos)

        unaccounted_atom_pos = atom_coords[unaccounted]

        unnaccounted_atom_nums = atomic_nums[unaccounted]

        bead_n += 1

    return beads, bead_connections, bead_n


def match_to_substructures(mol):

    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    # changes log for HAcceptorSmarts:
    #  v2, 1-Nov-2008, GL : fix amide-N exclusion; remove Fs from definition

    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0])]')

    #find any atoms that match these macs keys

    donor_atoms = [ i[0] for i in mol.GetSubstructMatches(HDonorSmarts, uniquify=1) ]

    acceptor_atoms = [ i[0] for i in mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1) ]

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

    return is_donor_atom,is_acceptor_atom


def make_representation(beads, m,bead_dist):

    atom_masses, atom_coords, atomic_nums,symbols,atom_aromatic = make_mol(m)

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

    ind1 = 0

    for b, ds in zip(beads, bead_dists_to_atoms):

        weights = 1 / (1 + np.exp(2 * (ds - bead_dist/2)))

        #weights = 1 / (1 + np.exp(  (ds -  bead_dist)))
        #make a vector of charges

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

        #aromatic

        Aromatic_vector = np.sum(weights * atom_aromatic)

        representation[ind1, 6] = Aromatic_vector

        # HDB

        HDB_vector = np.sum(weights * is_donor_atom)

        representation[ind1, 7] = HDB_vector

        # HBA

        HBA_vector = np.sum(weights * is_acceptor_atom)

        representation[ind1, 8] = HBA_vector

        ind1 += 1

    return np.array(representation)


def train_RF(train_descs,test_descs, train_IC50,test_IC50):

    train_fps = np.array([i for i in train_descs])

    test_fps = np.array(test_descs)

    train_y = np.array(train_IC50)

    test_y = np.array(test_IC50)

    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    rf.fit( train_fps,train_y)

    test_pred = rf.predict(test_fps)[0]

    return test_pred, test_y


def make_and_align_smiles(smiles):
    p = AllChem.ETKDGv2()
    mols = []

    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mol = standardize(mol)
        mols.append(mol)

    for mol in mols:
        mol.RemoveAllConformers()

    hmols_1_ = [Chem.AddHs(m) for m in mols]

    # Generate 100 conformers per each molecule
    for mol in hmols_1_:

        AllChem.EmbedMolecule(mol, p, )

        # AllChem.EmbedMultipleConfs(mol, 1, p)

        AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)


    hmols_1 = [Chem.RemoveHs(m) for m in hmols_1_]


    crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in hmols_1]


    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in hmols_1 ]

    sims = [ np.sum(DataStructs.BulkTanimotoSimilarity(fp,fps)) for fp in fps ]

    ref = np.argmax(sims)

    crippen_ref_contrib = crippen_contribs[ref]
    crippen_prob_contribs = crippen_contribs[:ref] + crippen_contribs[ref+1 :]

    ref_mol1 = hmols_1[ref]

    prob_mols_1 = hmols_1[:ref] + hmols_1[ref+1 :]


    crippen_score = []

    for idx, mol in enumerate(prob_mols_1):
        crippenO3A = rdMolAlign.GetCrippenO3A(mol, ref_mol1, crippen_prob_contribs[idx], crippen_ref_contrib, 0, 0)
        crippenO3A.Align()
        crippen_score.append(crippenO3A.Score())

    return hmols_1

def Physchem_calc(m):

    return np.array([Descriptors.MolWt(m), Crippen.MolLogP(m) , rdMolDescriptors.CalcNumHBA(m),rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcFractionCSP3(m)])

def Physchem_filter(prob_props,ref_props):

    mw_high = ref_props[0] + 100
    mw_low = ref_props[0] - 100

    logp_high = ref_props[1] + 1
    logp_low = ref_props[1] - 1

    f_sp3_high = ref_props[4] + 0.2
    f_sp3_low =ref_props[4] - 0.2

    return np.all(np.array([ prob_props[0] > mw_low, prob_props[0] < mw_high, prob_props[1] > logp_low, prob_props[1] < logp_high,prob_props[2] == ref_props[2] , prob_props[3] == ref_props[3] , prob_props[4] > f_sp3_low, prob_props[4] < f_sp3_high ]))


def make_graph(bead_n, bead_connections, encodings):

    G = nx.Graph()

    nodes = []

    for n in range(0, bead_n):
        nodes.append((n, {"encoding": encodings[0]}))

    G.add_nodes_from(nodes)
    G.add_edges_from(bead_connections)

    return G


###############


# make the reference molecule and representation

ref_mol = Chem.MolFromSmiles("O=C(NS(=O)(=O)C1C[C@@H]2CC[C@H]1C2)c1cc2ccccc2o1")
ref_mol = standardize(ref_mol)

#ref_mol = embed_mol_sdf(ref_mol)

ref_mol = embed_mol_sdf(ref_mol)

ref_beads, ref_bead_connections,bead_n = make_beads(ref_mol,R)

ref_rep = make_representation(ref_beads,ref_mol,R)

ref_encodings = bead_encode(ref_rep)

ref_G = make_graph(bead_n,ref_bead_connections,ref_encodings)

ref_hash = nx.weisfeiler_lehman_graph_hash(ref_G,node_attr='encoding')

ref_rep =make_representation(ref_beads,ref_mol,R)

##########

#physchem filters

ref_props = Physchem_calc(ref_mol)

##########

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

            m = Chem.MolFromSmiles(s)

            m = standardize(m)


        except:

            m = None

        if m:


            ###### calculate some physchem properties for initial filters

            prob_props = Physchem_calc(m)

            if Physchem_filter(prob_props, ref_props):

                ######

                # embed this new mol

                embed_mol_sdf(m)

                prob_beads, prob_bead_connections, prob_bead_n = make_beads(m, R)

                prob_rep = make_representation(prob_beads, m, R)

                prob_encodings = bead_encode(prob_rep)

                prob_G = make_graph(prob_bead_n, prob_bead_connections, prob_encodings)

                prob_hash = nx.weisfeiler_lehman_graph_hash(prob_G,node_attr='encoding')

                if prob_hash == ref_hash:

                    NNs.append(m)

                    print("found ", proc, len(NNs))

    if len(NNs) > 0:

        with Chem.SDWriter("output_" + str(proc) + ".sdf") as w:

            print("writing to "  "output_" + str(proc) + ".sdf", len(NNs), proc)

            for mol_ in NNs + [ref_mol]:
                w.write(mol_)

    else:

        print("none found " + str(proc))

db_length = len(open(database, "r").readlines())

inds = np.arange(0, db_length)

chunks = np.array_split(inds, maxproc)

args = []

c = 0

for i, j in enumerate(chunks):
    args.append((ref_mol, j, c))

    c += 1

# SearchWorker(args[0])

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

#SearchWorker((ref_mol, np.arange(0,10000) ,0  ))