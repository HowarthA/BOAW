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

# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")

# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def make_mol(mols,confIDs):

    all_coords = []
    all_masses = []
    all_aromatic = []
    all_atom_number =  []

    all_mol_inds = []
    all_conf_inds =[]

    all_atom_inds = []

    for m_id, molecule in enumerate( mols):

        for id in confIDs[m_id]:

            positions = molecule.GetConformer(id).GetPositions()

            a = 0

            for atom , p in zip( molecule.GetAtoms() , positions):

                all_masses.append(atom.GetMass())
                all_atom_number.append(atom.GetAtomicNum())
                all_aromatic.append(atom.GetIsAromatic())
                all_coords.append(p)
                all_mol_inds.append(m_id)
                all_conf_inds.append(id)
                all_atom_inds.append(a)

                a+=1

    return np.array(all_masses), np.array(all_coords), np.array(all_atom_number),np.array(all_aromatic),np.array(all_mol_inds),np.array(all_conf_inds),np.array(all_atom_inds)


def write_mol_xyz(mol,kmeans):

    f = open("suramin_p_jan.xyz","w")

    N_atoms=mol.GetNumAtoms()

    f.write(str(N_atoms+ len(kmeans) ) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(),mol.GetConformer(0).GetPositions()):

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

    ############next work out how many of the unnaccounted for atoms this bead now accounts for






    ############





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


def change_basis(mols ,confs, direction_vector,origin):

    for m,c_ids in zip(mols,confs):

        for c in c_ids:

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

    return mols


def make_beads(mols,confs):

    #pickle.dump(mols,open("mols.p","wb"))

    #pickle.dump(confs,open("confs.p","wb"))

    #### rotate mol onto PC axis

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    all_masses, all_coords, all_atom_number,all_aromatic,all_mol_inds,all_conf_inds,all_atom_inds = make_mol(mols, confs)

    #### rotate mol onto PC axis

    # atom_masses \= np.sum(atom_masses)

    origin = np.average(all_coords, axis=0, weights=all_masses)

    print("origin", origin)

    ###

    std_d = np.std(all_coords, axis=0)

    dist = 2 * np.sqrt(np.product(std_d[1:]))

    print("length scale", dist)

    #####

    # build to model by growing a web around the molecule

    # place initial bead

    closest_atom = np.argmin(np.linalg.norm(origin - all_coords, axis=1))

    beads = np.array([all_coords[closest_atom]])

    # need to count atoms that are accounted for by this bead

    atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in all_coords])


    #next find where the distance is less than the characterist length

    unaccounted_atom_coords = copy.copy(all_coords)
    unaccounted_atom_numbers = copy.copy(all_atom_number)

    unaccounted_mol_ids = copy.copy(all_mol_inds)
    unaccounted_conf_ids = copy.copy(all_conf_inds)
    unaccounted_atom_ids = copy.copy(all_atom_inds)


    accounted = np.min(atoms_dist_to_beads, axis=1) < dist/2

    accounted_mol_ids = all_mol_inds[accounted]
    accounted_conf_ids = all_conf_inds[accounted]
    accounted_atom_ids = all_atom_inds[accounted]

    #next find all of these atoms in all of the conformers of the molecules

    accounted_set = set([(m_i,a_i) for m_i,a_i in zip(  accounted_mol_ids,accounted_atom_ids)  ])

    print(len(accounted_set))

    for m_i,a_i in accounted_set:

        w = (unaccounted_atom_ids == a_i) * (unaccounted_mol_ids == m_i)

        unaccounted_atom_coords = unaccounted_atom_coords[~w]
        unaccounted_atom_numbers = unaccounted_atom_numbers[~w]

        unaccounted_mol_ids = unaccounted_mol_ids[~w]
        unaccounted_conf_ids = unaccounted_conf_ids[~w]
        unaccounted_atom_ids = unaccounted_atom_ids[~w]

    bead_n = 1

    connections = []

    while ( len(unaccounted_atom_coords) > 0 )  :

        #find which bead has the most atoms around it in a radius of 2*R>x>R

        accounted_ns = []

        new_beads_positions = []

        for  bead_for_connection,connection_bead_position in tqdm.tqdm(enumerate(beads)):

            #select the bead that has the most

            #print("connection bead",bead_for_connection)

            fit_params = Parameters()

            fit_params.add("p0", value=0, min=- np.pi , max=+ np.pi , vary=True)
            fit_params.add("p1", value=0, min=- np.pi , max=+ np.pi , vary=True)

            out = minimize(residual2, fit_params, args=(unaccounted_atom_numbers, unaccounted_atom_coords, beads,dist,connection_bead_position),
                       method='nelder',options={'maxiter':10000})

            #add this bead to the total

            initial_vector = basis[0]

            # next apply interpolated rotations

            p0 = out.params['p0']
            p1 = out.params['p1']

            r0_ab = Rotation.from_rotvec(-1 * basis[1] * p0)
            r1_ab = Rotation.from_rotvec(-1 * basis[2] * p1)

            initial_vector = r0_ab.apply(initial_vector)

            initial_vector = r1_ab.apply(initial_vector)

            new_bead_location = dist * initial_vector + connection_bead_position

            temp_beads = np.vstack((beads, new_bead_location))

            #next remove atoms that are now accounted for

            atoms_dist_to_beads = np.array([np.linalg.norm(b - temp_beads, axis=1) for b in unaccounted_atom_coords])

            accounted = np.min(atoms_dist_to_beads, axis=1) < dist / 2

            accounted_mol_ids = unaccounted_mol_ids[accounted]

            accounted_atom_ids = unaccounted_atom_ids[accounted]

            # next find all of these atoms in all of the conformers of the molecules

            accounted_set = set(accounted_mol_ids)

            accounted_ns.append(len(accounted_set))

            new_beads_positions.append(new_bead_location)

        ### decide on which connection lead to the greatest increase in new atoms accounted for

        #check if all the numbers of accounted atoms are the same if so break

        if (len(beads) > 2) & (np.all( accounted_ns == accounted_ns[0] )):

            break

        ###

        best = accounted_ns.index(max(accounted_ns))

        new_bead = new_beads_positions[best]

        beads = np.vstack((beads,new_bead))


        ############

        atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in all_coords])

        # next find where the distance is less than the characterist length

        unaccounted_atom_coords = copy.copy(all_coords)
        unaccounted_atom_numbers = copy.copy(all_atom_number)

        unaccounted_mol_ids = copy.copy(all_mol_inds)
        unaccounted_conf_ids = copy.copy(all_conf_inds)
        unaccounted_atom_ids = copy.copy(all_atom_inds)

        accounted = np.min(atoms_dist_to_beads, axis=1) < dist / 2

        accounted_mol_ids = all_mol_inds[accounted]
        accounted_conf_ids = all_conf_inds[accounted]
        accounted_atom_ids = all_atom_inds[accounted]

        # next find all of these atoms in all of the conformers of the molecules

        accounted_set = set([(m_i, a_i) for m_i, a_i in zip(accounted_mol_ids, accounted_atom_ids)])

        for m_i, a_i in accounted_set:

            w = (unaccounted_atom_ids == a_i) * (unaccounted_mol_ids == m_i)

            unaccounted_mol_ids = unaccounted_mol_ids[~w]
            unaccounted_conf_ids = unaccounted_conf_ids[~w]
            unaccounted_atom_ids = unaccounted_atom_ids[~w]

            unaccounted_atom_coords = unaccounted_atom_coords[~w]
            unaccounted_atom_numbers = unaccounted_atom_numbers[~w]

        ############

        bead_n +=1

        print("bead n" , bead_n , "atoms unaccounted",  len(unaccounted_atom_coords))

    print("n beads = " , bead_n)


    write_mol_xyz(mols[0],beads)

    '''
    G = nx.Graph()
    for c in connections:

        G.add_edge(c[0],c[1])

    nx.draw(G)
    '''

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


def make_representation(beads, mols,bead_dist):

    reps = []

    print("bead distance" , bead_dist)

    for m in mols:

        m = rdmolops.AddHs(m)

        atom_masses, atom_coords, atomic_nums,symbols,atom_aromatic = make_mol([m])

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

        #vectors_to_atoms = np.array([b - atom_coords for b in beads])
        #vectors_to_atoms /= np.linalg.norm(vectors_to_atoms, axis=2)[:, :, None]

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

            weights = 1 / (1 + np.exp( 2 * (ds - 2 * bead_dist)))

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

            HDA_vector = np.sum(weights * is_acceptor_atom)

            representation[ind1, 8] = HDA_vector

            ind1 += 1

        reps.append(representation.flatten())

    return np.array(reps)


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

    rot_bonds =  [Chem.rdMolDescriptors.CalcNumRotatableBonds(m) for m in mols]

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

        elif (rot_bonds[j]  >= 8) and (rot_bonds[j]  <=12 ):

            n_conformers = 200

        else:

            n_conformers = 300

        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

        confs.append([id for id in confIDs])

        AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters = 10000)

        h_mols.append(mol)

    crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in h_mols]

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in h_mols ]

    sims = [ np.sum(DataStructs.BulkTanimotoSimilarity(fp,fps)) for fp in fps ]

    ref = np.argmax(sims)

    crippen_ref_contrib = crippen_contribs[ref]
    crippen_prob_contribs = crippen_contribs[:ref] + crippen_contribs[ref+1 :]

    ref_mol1 = h_mols[ref]

    Chem.rdMolAlign.AlignMolConformers(ref_mol1)

    prob_mols_1 = h_mols[:ref] + h_mols[ref+1 :]

    crippen_score = []

    for idx, mol in enumerate(prob_mols_1):

        crippenO3A = rdMolAlign.GetCrippenO3A(mol, ref_mol1, crippen_prob_contribs[idx], crippen_ref_contrib, 0, 0)

        crippenO3A.Align()

        crippen_score.append(crippenO3A.Score())

        Chem.rdMolAlign.AlignMolConformers(mol)

    return h_mols , confs

#####

property = 'b_ratio'

property = "RPS"

df = pd.read_csv("/Users/alexanderhowarth/Desktop/total_b_ratio.csv").dropna(subset=property)

df = df.drop_duplicates(subset='Compound_ID')

print(len(df))

code = "TRB0005601 series"

df = df[df["STRUCTURE_COMMENT"] == code]

def MF_RF(ind1,df):

    IC50 = []
    smiles = []

    for i, r in tqdm.tqdm([ (i,r) for i,r in df.iterrows()]):

        print(r['Compound_ID'])

        IC50.append(r[property])

        smiles.append(r['Smiles'])

    #mols,confs = make_and_align_smiles(smiles)

    mols = pickle.load(open("mols.p","rb"))

    confs = pickle.load(open("confs.p","rb"))

    all_masses, all_coords, all_atom_number,all_aromatic,all_mol_inds,all_conf_inds,all_atom_inds = make_mol(mols, confs)

    direction_vector , origin = find_basis(all_coords,all_masses)

    mols = change_basis(mols,confs,direction_vector,origin)

    with Chem.SDWriter('align.sdf') as w:

        for m in mols:

            w.write(m)

    Rs = []

    av_vs = []

    av_ps = []

    std_ps = []

    for i in range(len(mols)):

        print("progress" , i/len(mols))

        train_IC50 = IC50[:i] + IC50[min(i + 1 ,len(IC50))  :]

        test_IC50 = IC50[i]

        train_mols = mols[:i] + mols[min(i + 1 ,len(mols)):]

        test_mols = mols[i]

        beads, bead_dist = make_beads(train_mols,confs)

        train_descs = make_representation(beads, train_mols,bead_dist)

        test_descs = make_representation(beads, [test_mols],bead_dist)

        test_vs = []

        test_ps = []

        for j in range(0, 2):

            # make predictions

            test_pred, test_val = train_RF(train_descs,test_descs, train_IC50,test_IC50)

            test_ps.append(test_pred)

            test_vs.append(test_val)

        std_ps.append(np.std(np.abs(np.array(test_vs) - np.array(test_ps))))
        av_vs.append(test_vs[0])
        av_ps.append(np.mean(test_ps, axis=0))

    r2 = r2_score(av_vs, av_ps)

    reg = LinearRegression().fit(np.array([[a] for a in av_vs]), av_ps)

    stddev_v = np.std(av_vs)

    rmse = mean_squared_error(av_vs, av_ps)

    rmse_med = mean_squared_error( av_vs , [ np.median(av_vs) for j in av_vs ] )

    std_errors = np.std([ abs(v-p) for v,p in zip(av_vs,av_ps) ] )

    plt.title("Beads on a string RF Model " + code + " n = " + str(len(IC50)) )
    plt.plot(av_vs, av_ps, "o", label="R2 = " + str( round(r2,2)) + "\nstd = " + str(round(std_errors,2)) + "\nRMSE = " +str( round( rmse,2))+ "\nRMSE/stddev(values) = " + str( round( rmse/stddev_v,2)) + "\nRMSE/RMSE(no skill model) = " + str( round( rmse/rmse_med,2)) ,alpha =0.8)
    plt.plot([min(av_vs), max(av_vs)], [min(av_vs), max(av_vs)], linestyle=":",color = 'grey')

    plt.plot([min(av_vs), max(av_vs)], [reg.coef_*min(av_vs) + reg.intercept_, reg.coef_*max(av_vs) + reg.intercept_],color = "C0")

    plt.legend(

    )

    plt.xlabel("Experimental")
    plt.ylabel("Predicted")

    #plt.savefig(folder + "/" + r['ID'] + ".png")

    plt.show()
    plt.close()

MF_RF(0,df)



