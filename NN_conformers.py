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



# df = pd.read_csv("/Users/alexanderhowarth/Documents/G3BP1/TRB000"+str(code)+"/G3BP1_"+str(code)+"_grouped.csv")
# df['suramin_normalised_mean'] = np.abs(df['suramin_normalised_mean'])

basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def embed_mol_sdf(mol,conf_n):

    if mol:

        rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

        mol = Chem.AddHs(mol)

        # Generate conformers

        if rot_bonds <8:

            n_conformers = 50

        elif (rot_bonds >=8) and (rot_bonds <=12 ):

            n_conformers = 200

        else:

            n_conformers = 300


        confIDs = AllChem.EmbedMultipleConfs(mol, n_conformers)

        AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters = 10000)

        #mol = rdmolops.RemoveHs(mol)

        return (mol , confIDs)

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

        '''
        if rot_bonds <8:

            n_conformers = 50

        elif (rot_bonds >=8) and (rot_bonds <=12 ):

            n_conformers = 200

        else:

            n_conformers = 300
        '''

        confIDs = AllChem.EmbedMultipleConfs(mol, 1)

        AllChem.MMFFOptimizeMoleculeConfs(mol,maxIters = 10000)

        #AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)

        #mol = rdmolops.RemoveHs(mol)


        return (mol , confIDs)

    else:

        return None


def make_mol(molecule,conf_ID):

    atomic_mass = []
    atomic_numbers = []
    atom_aromatic = []

    positions = molecule.GetConformer(conf_ID).GetPositions()

    for atom in molecule.GetAtoms():
        atomic_mass.append(atom.GetMass())
        atomic_numbers.append(atom.GetAtomicNum())
        atom_aromatic.append(atom.GetIsAromatic())


    return np.array(atomic_mass), np.array(positions), np.array(atomic_numbers), np.array(atom_aromatic)


def write_mols_xyz(mols, kmeans):
    f = open("ref_mol.xyz", "w")

    N_atoms = 0
    for m in mols:
        N_atoms += m.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for m in mols:

        p = m.GetConformer(0).GetPositions()

        for atom, coords in zip(m.GetAtoms(), p):
            a = atom.GetSymbol()

            f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def write_mol_xyz(mol, coords_, kmeans):
    f = open("ref_mol.xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), coords_):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def write_mol_vector(mol, coords_, kmeans, vectors):
    f = open("test_mol_5601.xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms + len(kmeans)) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), coords_):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    for coords in kmeans:
        f.write(str("Xe") + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    # for coords,vec in zip(kmeans,vectors):

    #   print(vec)

    #  f.write(str("Ar") + " " + str(coords[0] + vec[0]) + " " + str(coords[1]+ vec[1]) + " " + str(coords[2]+ vec[2]) + "\n")

    f.close()


def write_xyz(mol, coords_):
    f = open("suramin_p_.xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), coords_):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def xyz(mol, pos):
    f = open("suramin_rot_pos_.xyz", "w")

    N_atoms = mol.GetNumAtoms()

    f.write(str(N_atoms) + "\n" + "\n")

    for atom, coords in zip(mol.GetAtoms(), pos):
        a = atom.GetSymbol()

        f.write(a + " " + str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2]) + "\n")

    f.close()


def residual(params, weights, pos, origin):
    p1 = params['p1']
    p2 = params['p2']
    R = params['R']

    c_ = R * np.array([np.sin(p1) * np.cos(p2), np.sin(p1) * np.sin(p2), np.cos(p1)]) + origin

    ##### next calculate weighted distance to the atoms

    ds = 1 - np.exp(- np.linalg.norm(c_ - pos, axis=1) / 5 * weights)

    return np.sum(ds)


def residual2(params, atomic_nums, unaccounted_atom_pos, beads, dist, connection_bead_position):
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

    # next work out how many of the unnaccounted for atoms this bead now accounts for

    dists = np.array([np.linalg.norm(new_bead_location - p) for p in unaccounted_atom_pos])

    scores = 1 / (1 + np.exp(-2 * (dists - dist / 2))) * atomic_nums

    return np.sum(scores)


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


def make_representation(beads, m,bead_dist,conf_id):

    atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m,conf_id)

    is_donor_atom, is_acceptor_atom = match_to_substructures(m)

    ############# next job is to "color the beads in"

    # define some atomwise properties

    ComputeGasteigerCharges(m)

    charges = []

    for a in m.GetAtoms():
        charges.append(abs(float(a.GetProp("_GasteigerCharge"))))

    from rdkit.Chem.Draw import SimilarityMaps
    from rdkit.Chem import Draw

    charges = np.array(charges)

    CC = np.array(rdMolDescriptors._CalcCrippenContribs(m))

    ASA = np.array([k for k in rdMolDescriptors._CalcLabuteASAContribs(m)[0]])

    TPSA = np.array(rdMolDescriptors._CalcTPSAContribs(m))

    logP_c = CC[:, 0]

    MR_c = CC[:, 1]

    '''

    d2d = Draw.MolDraw2DSVG(400,400,400,400)

    SimilarityMaps.GetSimilarityMapFromWeights(m,[p for p in logP_c],draw2d=d2d, colorMap='jet', contourLines=10)

    d2d.FinishDrawing()

    pic = open("test_5601_charges.svg", "w+")

    pic.write(str(d2d.GetDrawingText()))

    pic.close()

    quit()

    '''

    ###### next make the representation

    representation = np.zeros((len(beads), 9))

    # find distances to atoms from beads

    bead_dists_to_atoms = np.array([np.linalg.norm(b - atom_coords, axis=1) for b in beads])

    # find normalised vectors to atoms from beads

    # vectors_to_atoms = np.array([b - atom_coords for b in beads])

    # vectors_to_atoms /= np.linalg.norm(vectors_to_atoms, axis=2)[:, :, None]

    ind1 = 0

    for b, ds in zip(beads, bead_dists_to_atoms):
        weights = 1 / (1 + np.exp((ds - bead_dist / 2)))

        # make a vector of charges
        charge_vector = np.sum(weights * charges)

        representation[ind1, 0] = charge_vector

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

        Aromatic_vector = np.sum(weights * atom_aromatic)

        representation[ind1, 6] = Aromatic_vector + 1

        # HDB

        HDB_vector = np.sum(weights * is_donor_atom)

        representation[ind1, 7] = HDB_vector + 1

        # HBA

        HDA_vector = np.sum(weights * is_acceptor_atom)

        representation[ind1, 8] = HDA_vector + 1

        ind1 += 1

    return representation


def find_basis(coordinates, masses):

    origin = np.average(coordinates, axis=0, weights=masses)

    coordinates -= origin

    pca = decomposition.PCA(n_components=3)

    pca.fit(coordinates)

    direction_vector = pca.components_

    return  direction_vector , origin


def change_basis(m , direction_vector,origin,conf_ID):

    conf = m.GetConformer(conf_ID)

    coordinates = conf.GetPositions()

    a_c = []

    for a_ in coordinates:

        a_ = np.matmul(direction_vector, a_)

        a_c.append(a_)

    atom_coords = np.array(a_c)

    atom_coords -= origin


    for i in range(m.GetNumAtoms()):
        x, y, z = atom_coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))


def make_beads(m,confIDs):

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    t_beads = []

    t_dist = []

    #### rotate mol onto PC axis

    for i in confIDs:

        print("conf id" , i )

        atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m,i)

        match_to_substructures(m)

        # atom_masses \= np.sum(atom_masses)

        origin = np.average(atom_coords, axis=0, weights=atom_masses)

        print("origin", origin)

        ###

        std_d = np.std(atom_coords, axis=0)

        dist = 4 * np.sqrt(np.product(std_d[1:]))

        print("dist", dist)

        #####

        # build to model by growing a web around the molecule

        # place initial bead
        closest_atom = np.argmin(np.linalg.norm(origin - atom_coords, axis=1))

        beads = np.array([atom_coords[closest_atom]])

        # need to count atoms that are accounted for by this bead

        atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in atom_coords])

        unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist / 2

        unaccounted_atom_pos = atom_coords[unaccounted]

        unaccounted_atom_pos_old = len(unaccounted_atom_pos) +1

        unnaccounted_atom_nums = atomic_nums[unaccounted]

        bead_n = 0

        while len(unaccounted_atom_pos) > 0 & (len(unaccounted_atom_pos) < unaccounted_atom_pos_old):

            # find which bead has the most atoms around it in a radius of 2*R>x>R

            unnaccounted_ns = []

            new_beads_positions = []

            for bead_for_connection, connection_bead_position in enumerate(beads):
                # select the bead that has the most

                fit_params = Parameters()

                fit_params.add("p0", value=0, min=- np.pi , max=+ np.pi , vary=True)
                fit_params.add("p1", value=0, min=- np.pi , max=+ np.pi , vary=True)

                out = minimize(residual2, fit_params,
                               args=(unnaccounted_atom_nums, unaccounted_atom_pos, beads, dist, connection_bead_position),
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

                temp_beads = np.vstack((beads, new_bead_location))

                # next remove atoms that are now accounted for

                atoms_dist_to_beads = np.array([np.linalg.norm(b - temp_beads, axis=1) for b in unaccounted_atom_pos])

                unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist / 2

                unnaccounted_ns.append(np.sum(unaccounted))

                new_beads_positions.append(new_bead_location)

            ### decide on which connection lead to the greatest increase in new atoms accounted for

            if (len(beads) > 2) & (np.all(unnaccounted_ns == unnaccounted_ns[0])):
                break

            best = unnaccounted_ns.index(min(unnaccounted_ns))

            new_bead = new_beads_positions[best]

            beads = np.vstack((beads, new_bead))

            atoms_dist_to_beads = np.array([np.linalg.norm(b - beads, axis=1) for b in unaccounted_atom_pos])

            unaccounted = np.min(atoms_dist_to_beads, axis=1) > dist

            unaccounted_atom_pos_old = len(unaccounted_atom_pos)

            unaccounted_atom_pos = unaccounted_atom_pos[unaccounted]

            unnaccounted_atom_nums = unnaccounted_atom_nums[unaccounted]

            bead_n += 1

        t_dist.append(dist)

        t_beads.append(beads)

        #write_mol_xyz(m, atom_coords, beads)

    return t_beads, t_dist


def align_mols(ref, crippen_ref_contrib, prob,ref_confIDs,prob_confIDs):

    crippen_prob_contrib = rdMolDescriptors._CalcCrippenContribs(prob)

    # pick which mol should be centre of the group

    score = 0

    rc_pc = [0,0]

    for pi in prob_confIDs:

        crippenO3A = rdMolAlign.GetCrippenO3A(prob, ref, crippen_prob_contrib, crippen_ref_contrib, pi, 0)

        crippenO3A.Align()

        score_ = crippenO3A.Score()

        if score_ > score:

            score = float(score_)

            rc_pc[0] = 0
            rc_pc[1] = pi

    return rc_pc


def Physchem_calc(m):

    return np.array([Descriptors.MolWt(m), Crippen.MolLogP(m) , rdMolDescriptors.CalcNumHBA(m),rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcFractionCSP3(m)])

def Physchem_filter(prob_props,ref_props):

    mw_high = ref_props[0] + 50
    mw_low = ref_props[0] - 50

    logp_high = ref_props[1] +1
    logp_low = ref_props[1] -1

    f_sp3_high = ref_props[4] + 0.2
    f_sp3_low =ref_props[4] - 0.2

    return np.all(np.array([ prob_props[0] > mw_low, prob_props[0] < mw_high, prob_props[1] > logp_low, prob_props[1] < logp_high,prob_props[2] == ref_props[2] , prob_props[3] == ref_props[3] , prob_props[4] > f_sp3_low, prob_props[4] < f_sp3_high ]))


# make the reference molecule and representation

ref_mol, ref_confIDs = embed_mol_smiles("O=C(NS(=O)(=O)C1C[C@@H]2CC[C@H]1C2)c1cc2ccccc2o1")


#ref_mol, ref_confIDs = embed_mol_smiles("Cc1ccc(OCC(=O)Nn2c(-c3ccco3)n[nH]c2=S)c(C)c1")


##########

#physchem filters

ref_props = Physchem_calc(ref_mol)

##########

#ref_mol, ref_confIDs = embed_mol_smiles("CN(C)CCCN(C(=O)c1cccc(C(F)(F)F)c1)c1nc2ccc(F)cc2s1")

for i in ref_confIDs:

    atomic_mass, positions, atomic_numbers, atom_aromatic = make_mol(ref_mol,i)

    direction_vector, origin = find_basis(positions, atomic_mass)

    change_basis(ref_mol, direction_vector, origin,i)

database = os.path.expanduser("~/mcule_purchasable_in_stock_221205.smi")

#database = "/Users/alexanderhowarth/Downloads/mcule_purchasable_in_stock_221205.smi"

#database = "5601.smi"

db_length = len(open(database, "r").readlines())

beads, bead_dist = make_beads(ref_mol,ref_confIDs)

ref_rep = []

for beads_,bead_dist_,confid in zip(beads,bead_dist,ref_confIDs):

    ref_rep_ = make_representation(beads_, ref_mol,bead_dist_,confid)

    ref_rep.append(ref_rep_)

crippen_ref_contrib = rdMolDescriptors._CalcCrippenContribs(ref_mol)

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

            m = Chem.MolFromSmiles(s)

            m = standardize(m)


        except:

            m = None


        if m:

            ###### calculate some physchem properties for initial filters

            prob_props = Physchem_calc(m)

            if Physchem_filter(prob_props,ref_props):

            ######

            # embed this new mol

                embed = embed_mol_sdf(m,1)

            else:

                embed = None

            if embed:

                prob_mol = embed[0]

                prob_confIDs = embed[1]

                '''
                for c in prob_confIDs:
    
                    atomic_mass, positions, atomic_numbers, atom_aromatic = make_mol(prob_mol, c)
    
                    direction_vector, origin = find_basis(positions, atomic_mass)
    
                    change_basis(prob_mol, direction_vector, origin, c)
                '''

                # align the mol against the template

                rc_pc = align_mols(ref_mol, crippen_ref_contrib, prob_mol,ref_confIDs,prob_confIDs)

                # generate the represenation of the prob molecule utilising the reference grid

                prob_rep = make_representation(beads[rc_pc[0]], prob_mol,bead_dist[rc_pc[0]],rc_pc[1])

                # measure the similarity

                rep_fraction = 2 * np.abs(prob_rep - ref_rep[rc_pc[0]]) / (prob_rep + ref_rep[rc_pc[0]])

                if np.all(rep_fraction < threshold):

                    prob_mol.SetProp("_similarity", str(np.median(rep_fraction)))

                    NNs.append(prob_mol)

                    print("found " , proc ,len(NNs))

                    #with Chem.SDWriter(output_folder + "/match_" + str(uuid.uuid4()) + ".sdf") as w:

                        #for m1 in [ref_mol, prob_mol]:

                            #w.write(m1)

    if len(NNs) > 0:

        with Chem.SDWriter( "output_" + str(proc) + ".sdf" ) as w:

            print("writing to "  "output_" + str(proc) + ".sdf" , len(NNs) , proc )

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

    c +=1


#SearchWorker(args[0])


import multiprocessing

p = multiprocessing.Pool()

# defaults to os.cpu_count() workers
p.map_async(SearchWorker,  args )

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