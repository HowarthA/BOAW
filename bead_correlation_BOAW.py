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

# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")

# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

colors = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.4, 0.74),
          (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81)]

R = 2.1943998623787615 * 2


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def embed_mol_sdf(mol, conf_n):
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

        # mol = rdmolops.RemoveHs(mol)

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

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

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


def write_mol_xyz(mol, kmeans,title):
    f = open( title, "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), mol.GetConformer(0).GetPositions()):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


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

            print("loop " , bead_n)

            # find which bead has the most atoms around it in a radius of 2*R>x>R

            accounted_ns = []

            new_beads_positions = []

            for bead_for_connection, connection_bead_position in enumerate(beads):

                print("bead connection" , bead_for_connection)

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

                print(atoms_dist_to_beads)

                accounted = atoms_dist_to_beads < dist

                print("accouted" , np.sum(accounted))

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

        write_mol_xyz(m , beads,"mol_" + str(counter) + ".xyz")

        t_beads.append(beads)

    return t_beads


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

    print("bead distance", bead_dist)

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


def train_RF(train_descs, test_descs, train_IC50, test_IC50):
    train_fps = np.array([i for i in train_descs])

    test_fps = np.array(test_descs)

    train_y = np.array(train_IC50)

    test_y = np.array(test_IC50)

    rf = RandomForestRegressor(n_estimators=10, random_state=42)

    rf.fit(train_fps, train_y)

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

    rot_bonds = [Chem.rdMolDescriptors.CalcNumRotatableBonds(m) for m in mols]

    hmols_1_ = [Chem.AddHs(m) for m in mols]

    confs = []

    h_mols = []

    print("embedding mols")

    # Generate 100 conformers per each molecule

    for j, mol in tqdm.tqdm(enumerate(hmols_1_)):

        mol = Chem.AddHs(mol)

        # Generate conformers

        if rot_bonds[j] < 8:

            n_conformers = 50

        elif (rot_bonds[j] >= 8) and (rot_bonds[j] <= 12):

            n_conformers = 200

        else:

            n_conformers = 300

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

        confs.append([id for id in confIDs])

        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

        h_mols.append(mol)

    crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in h_mols]

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in h_mols]

    sims = [np.sum(DataStructs.BulkTanimotoSimilarity(fp, fps)) for fp in fps]

    ref = np.argmax(sims)

    crippen_ref_contrib = crippen_contribs[ref]
    crippen_prob_contribs = crippen_contribs[:ref] + crippen_contribs[ref + 1:]

    ref_mol1 = h_mols[ref]

    Chem.rdMolAlign.AlignMolConformers(ref_mol1)

    prob_mols_1 = h_mols[:ref] + h_mols[ref + 1:]

    crippen_score = []

    for idx, mol in enumerate(prob_mols_1):
        crippenO3A = rdMolAlign.GetCrippenO3A(mol, ref_mol1, crippen_prob_contribs[idx], crippen_ref_contrib, 0, 0)

        crippenO3A.Align()

        crippen_score.append(crippenO3A.Score())

        Chem.rdMolAlign.AlignMolConformers(mol)

    return h_mols, confs


#####

property = 'b_ratio'

df = pd.read_csv("/Users/alexanderhowarth/Desktop/total_b_ratio.csv").dropna(subset=property)

df = df.drop_duplicates(subset='Compound_ID')

code = "TRB0005601 series"

df = df[df["STRUCTURE_COMMENT"] == code]



smiles = []

for i, r in tqdm.tqdm([(i, r) for i, r in df.iterrows()]):
    print(r['Compound_ID'])

    smiles.append(r['Smiles'])

mols = []

confs = []

for s in tqdm.tqdm(smiles):

    mol , confIDs = embed_mol_smiles(s)

    mols.append(mol)

    confs.append([ i for i in confIDs])


for i, m in enumerate(mols):

    all_masses, all_coords, all_atom_number, all_aromatic = make_mol(m, confs[i])

    direction_vector, origin = find_basis(all_coords, all_masses)

    change_basis(m, confs[i], direction_vector, origin)

# next make beads and all representations for all conformers and mols

total_beads = []

total_reps = [ ]

for i, m in tqdm.tqdm(enumerate(mols)):

    beads = make_beads(m, confs[i], R,i)

    rep = make_representation(beads,m,R,confs[i])

    total_beads.append(beads)
    total_reps.append(rep)




pickle.dump(mols, open("mols_cb.p", "wb"))
pickle.dump(total_reps, open("total_reps.p", "wb"))
pickle.dump(total_beads, open("total_beads.p", "wb"))
#total_beads = pickle.load(open("total_beads.p", "rb"))

quit()

print("mols" , len(mols))
print("beads", np.shape(total_beads))

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

def normalise_rep(rep):

    rep = rep - np.mean(rep , axis = 0)

    stdev = np.std( rep,axis=0 ) + 0.000001

    rep /= stdev

    return rep

def residual_worker(params,coords_1,coords_2):

    #next apply interpolated rotations

    p0 = params['p0']
    p1 = params['p1']

    x = params['x']
    y = params['y']
    z = params['z']

    coords_2 = transform(coords_2,p0,p1,x,y,z)

    return np.sum(np.linalg.norm( coords_1 - coords_2 ,axis = 1))

def allign_reps(beads_1,beads_2,rep1,rep2):

    rep1 = normalise_rep(rep1)

    rep2 = normalise_rep(rep2)

    ds = np.array([np.linalg.norm( rep2 - c , axis = 1 ) for c in rep1 ])

    r1_a, r2_a = linear_sum_assignment(ds)

    beads_1_a = beads_1[r1_a]

    beads_2_a = beads_2[r2_a]

    #print([ ds[c, r ] for c,r in zip(col_ind,row_ind)  ])

    fit_params = Parameters()

    fit_params.add("p0", value=0, min=0, max= 2 * np.pi, vary=True)
    fit_params.add("p1", value=0, min= 0, max=2 * np.pi, vary=True)

    fit_params.add("x", value=0,  vary=True)
    fit_params.add("y", value=0, vary=True)
    fit_params.add("z", value=0, vary=True)

    #write_xyz(beads_1,beads_2)

    out = minimize( residual_worker, fit_params,
                   args=(beads_1_a,beads_2_a),
                    method='nelder' )

    aligned_beads_2 = transform(beads_2,out.params['p0'],out.params['p1'],out.params['x'],out.params['y'],out.params['z'])

    return aligned_beads_2


#next need to make the correlation matrix for all of the beads that have been fitted.

total_n = 0

print(len(total_beads))

all_mol_inds= []

all_bead_inds = []

bead_ind = []

m = 0

total_n = 0

for bs in total_beads:

    for b_ in range(len(bs[0])):

        all_mol_inds.append(m)

        all_bead_inds.append(b_)

        bead_ind.append(total_n)

        total_n += 1

    m+=1

print("total beads" , total_n )

index_vector = np.array([ [bn , mi, bi] for bn,mi,bi in zip( bead_ind,all_mol_inds,all_bead_inds  ) ])

print(index_vector)

correlation_M = np.zeros(( total_n  , total_n))

print("corr" , np.shape(correlation_M))

i_bead_n = 0

'''
for bs_i, rs_i in zip(total_beads, total_reps):

    for c_b_i, c_r_i in zip(bs_i, rs_i):

        for bs_j, rs_j in zip(total_beads, total_reps):

            for c_b_j, c_r_j in zip(bs_j, rs_j):

                aligned_c_b_j = allign_reps(c_b_i,c_b_j,c_r_i,c_r_j)

                for j_counter, c__ in enumerate(aligned_c_b_j):

                    ds =  np.linalg.norm(c_b_i -  c__, axis=1 )

                    ds_ = ds < R/2

                    for i_counter , v in enumerate(ds_):

                        print( "i" ,i_bead_n + i_counter, "j" ,j_bead_n + j_counter )

                        correlation_M[ i_bead_n + i_counter, j_bead_n + j_counter ] +=v

            j_bead_n += len(bs_j[0])

    i_bead_n += len(bs_i[0])
'''


for bs_i, rs_i in tqdm.tqdm(zip(total_beads, total_reps)):

    c_b_i = bs_i[0]
    c_r_i = rs_i[0]

    j_bead_n = 0

    for bs_j, rs_j in zip(total_beads, total_reps):

        c_b_j = bs_j[0]
        c_r_j = rs_j[0]

        aligned_c_b_j = allign_reps(c_b_i,c_b_j,c_r_i,c_r_j)

        for j_counter, c__ in enumerate(aligned_c_b_j):

            ds =  np.linalg.norm(c_b_i -  c__, axis=1 )

            ds_ = ds < R/2

            for i_counter , v in enumerate(ds_):

                #print( "i" ,i_bead_n + i_counter, "j" ,j_bead_n + j_counter )

                correlation_M[ i_bead_n + i_counter, j_bead_n + j_counter ] +=v

        j_bead_n += len(c_b_j)

    i_bead_n += len(c_b_i)


#pickle.dump(correlation_M,open("corrM.p","wb"))

C_matrix = pickle.load(open("corrM.p","rb"))

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

# Ensure the correlation matrix is symmetric
corr = (C_matrix + C_matrix.T) / 2
np.fill_diagonal(corr, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)

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

# how many clusters do we need before the cluster labels appear in each molecule only once

cluster_ids = hierarchy.fcluster(dist_linkage, 1.2, criterion="distance")
'''
for m_id in range(0,len(mols)):

    w  = np.where(index_vector[:,1] == m_id)[0]

    c_ids = cluster_ids[w]

    if len(c_ids) != len(set(c_ids)):

        print(c_ids)

        print("try again")

        break

quit()
'''
print("n clusters", len(set(cluster_ids)))

'''
for _cluster_id in set(cluster_ids):

    w_c = np.where(cluster_ids == _cluster_id)

    cluster_pos = all_coords[w_c]

    dmins = np.max(cluster_pos, axis=0) - np.min(cluster_pos, axis=0)

    v = np.product(dmins)

    r = np.cbrt(3 / (4 * np.pi) * v)

    print("r = ", r)
'''

atom_set = []

color_set = []

for m_n,m in enumerate(mols):

    w_m2 = np.where(index_vector[:, 1] == m_n)

    c_ids2 = cluster_ids[w_m2]

    m_beads = total_beads[m_n][0]

    AllChem.Compute2DCoords(m)

    atom_set.append([])
    color_set.append({})

    for a , p in enumerate(m.GetConformer(0).GetPositions()):

        #find closest bead

        b_ind = np.argmin( np.linalg.norm(p - m_beads , axis= 1) )

        c = c_ids2[b_ind]

        color_set[-1][a ] = copy.copy(colors[c % len(colors)])

        atom_set[-1].append(a )

######

n_per_row = 10

scale = 300

n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

img = d2d.DrawMolecules(list(mols), highlightAtoms=atom_set, highlightAtomColors=color_set)

pic = open("clustered.svg", "w+")

d2d.FinishDrawing()

pic.write(str(d2d.GetDrawingText()))

pic.close()

