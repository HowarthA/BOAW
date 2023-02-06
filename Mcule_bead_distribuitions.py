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
import os
from rdkit.Chem import Descriptors
# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")

# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

R = 2.1943998623787615 * 2

maxproc = 600

def embed_mol_sdf(mol):

    mol = rdmolops.AddHs(mol)

    #AllChem.EmbedMolecule(mol)



    try:


        converge = AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

    except:

        return None

    if converge != -1:

        return mol

    else:

        return None


def embed_mol_smiles(s):

    mol = Chem.MolFromSmiles(s)
    mol = standardize(mol)
    mol = rdmolops.AddHs(mol)

    if mol:

        mol = Chem.AddHs(mol)

        # Generate conformers

        '''
        if rot_bonds <8:

            n_conformers = 50

        elif (rot_bonds >=8) and (rot_bonds <=12 ):

            n_conformers = 200

        else:

            n_conformers = 300
        '''

        AllChem.EmbedMolecule(mol)

        converge = AllChem.MMFFOptimizeMolecule(mol,maxIters = 10000)

        if converge != -1:

            return mol

        else:

            return None


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def make_mol(mol):

    atomic_mass = []
    atomic_numbers = []
    atom_aromatic = []

    symbols = []

    positions = mol.GetConformer().GetPositions()

    for atom in mol.GetAtoms():

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

    bead_n = 0

    while len(unaccounted_atom_pos) > 0 & (len(unaccounted_atom_pos) < unaccounted_atom_pos_old):

        # find which bead has the most atoms around it in a radius of 2*R>x>R

        accounted_ns = []

        new_beads_positions = []

        for bead_for_connection, connection_bead_position in enumerate(beads):

            # select the bead that has the most

            fit_params = Parameters()

            fit_params.add("p0", value=0, min=- np.pi, max=+ np.pi, vary=True)
            fit_params.add("p1", value=0, min= - np.pi/2, max= np.pi/2, vary=True)

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

        if (len(beads) > 2) & (sum(accounted_ns) == 0):

            break

        best = accounted_ns.index(max(accounted_ns))

        new_bead = new_beads_positions[best]

        beads = np.vstack((beads, new_bead))

        atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in atom_coords])

        unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist

        unaccounted_atom_pos_old = len(unaccounted_atom_pos)

        unaccounted_atom_pos = atom_coords[unaccounted]

        unnaccounted_atom_nums = atomic_nums[unaccounted]

        bead_n += 1

    return beads, dist

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

    # find normalised vectors to atoms from beads

    #vectors_to_atoms = np.array([b - atom_coords for b in beads])
    #vectors_to_atoms /= np.linalg.norm(vectors_to_atoms, axis=2)[:, :, None]

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

    return representation

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


#####
#
database = os.path.expanduser("~/mcule_purchasable_in_stock_221205.smi")

#database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"


def Search(args):

    i = 0

    inds = args[0]

    proc = args[1]

    s_file = open(database, "r")

    total_beads = []

    while i < inds[0] - 1:

        s_file.readline()

        i += 1

    for i in tqdm.tqdm(inds):

        s = s_file.readline()

        s = s.split()[0]

        try:
            mol = Chem.MolFromSmiles(s)

            mol = standardize(mol)

            if (mol.GetNumAtoms() < 2) or (Descriptors.MolWt(mol) > 1000):

                mol = None

        except:

            mol = None


        if mol:

            mol = embed_mol_sdf(mol)

        else:

            embed = None

        if mol:

            #atomic_mass , positions, atomic_numbers, symbols, atom_aromatic = make_mol([mol])

            #direction_vector , origin = find_basis(positions,atomic_mass)

            #mols = change_basis([mol],direction_vector,origin)

            beads_, bead_dist  = make_beads(mol,R)

            train_descs = make_representation(beads_, mol,R)

            total_beads.append([ r for r in  train_descs[0]])

            #now look at kde of columns of this matrix

    titles = ["abs charge" , "masses","logP","MrC","ASA","TPSA","Aromatic","HBD","HBA"]

    total_beads = np.array(total_beads)

    pickle.dump(total_beads, open("beads_proc" + str(proc) + ".p", "wb"))

    #pickle.dump(boundaries, open("mcule_sample_boundaries.p","wb"))

    '''
    bead_encoding = []


    for bead in total_beads:

        encoding = [0 for b in range(np.shape(total_beads)[1]) ]

        for b_id , p in enumerate(bead):

            w_b = np.argmin( np.abs(p - boundaries[b_id] ))

            encoding[b_id] = int(w_b)

        bead_encoding.append(tuple(encoding))

    print(len(set(bead_encoding)))

    print(np.array(bead_encoding))
    '''


def estimate_R(s_file, n_samples):

    Rs = []

    for ind in tqdm.tqdm(range(0, n_samples)):

        s = s_file.readline()

        s = s.split()[0]

        try:
            mol,conf = embed_mol_smiles(s)

            Rs.append(Descriptors3D.RadiusOfGyration(mol )/ 2)


        except:

            None

    return np.average(np.array(Rs)[~np.isnan(Rs)])

db_length = len(open(database, "r").readlines())

inds = np.arange(0, db_length)

chunks = np.array_split(inds, maxproc)

args = []

c = 0

for i, j in enumerate(chunks):

    args.append(( j, c))

    c +=1

import multiprocessing

p = multiprocessing.Pool()

# defaults to os.cpu_count() workers
p.map_async(Search,  args )

# perform process for each i in i_list
p.close()
p.join()

#first work out average distance:

#R = estimate_R(db_,n_samples)



