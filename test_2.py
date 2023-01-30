from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import tqdm
import pandas as pd
import pickle
import numpy as np

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


target_selection = pd.read_csv("/Users/alexanderhowarth/PycharmProjects/Library_Analysis/Clustering_files/all_compounds_with_origin.csv")


target_inchi_key = []

target_selection = target_selection[ (target_selection['library'] == 'PPI_Library.sdf') | ( target_selection['library'] == 'Protein_Mimetics_Library.sdf')]

for i,r in tqdm.tqdm(target_selection.iterrows()):

    try:

        target_inchi_key.append( [Chem.MolToInchiKey(standardize(Chem.MolFromSmiles(r['smiles']))) , r['origin'] , r['library'] ] )

    except:

        print("Broken Mol ")

target_inchi_key = np.array(target_inchi_key)

pickle.dump(target_inchi_key,open("target_key","wb"))



target_inchi_key = pickle.load(open("target_key","rb"))

#pickle.dump(LT1, open("LT1.p","wb"))


#LT1 = pickle.load(open("LT1.p","rb"))

found_mols = []

for m in  tqdm.tqdm( Chem.SDMolSupplier("/Users/alexanderhowarth/Downloads/sdfbrowserexport-FRUIF9UE.sdf") ):

    m = standardize(m)

    ID = m.GetProp('ID')

    inchi_key = Chem.MolToInchiKey(m)

    w = np.where(target_inchi_key[:,0] == inchi_key)[0]

    if len(w) >0 :

        found_mols.append([ID , target_inchi_key[w[0],1] , inchi_key, target_inchi_key[w[0],2]])

    else:
        found_mols.append([ID , 'not found',inchi_key])

pickle.dump(found_mols,open("PPI_TRB.p","wb"))


#target_inchi_key = pickle.load(open("target_key","rb"))



found = pickle.load(open("PPI_TRB.p","rb"))

df = pd.DataFrame(found,columns=['TRB','origin','Inchi','library']).drop_duplicates(subset = 'TRB')


print(len(df))

df = df[df['origin'] != 'not found']

print(df)

for i ,r in df.iterrows():

    if r['origin'] == 'nan':

        r['origin'] = 'diversity'

print(df)

df.to_csv("TRB_PPI.csv")

df2 = pd.read_csv("/Users/alexanderhowarth/Desktop/Targeted_library_25k.csv")

df2_inchi_key = [ i for i in df2['inchi_key']]

print(df2_inchi_key)

c = 0

for i in df['Inchi']:

    if i in df2_inchi_key:

        c+=1

print(c)