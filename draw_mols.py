from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import numpy as np

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

mols = []
inchi_keys = set()
similarity = []

for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/TRB705_morfNN/NNs_.sdf"):

    m = standardize(m)

    if Chem.MolToInchiKey(m) not in inchi_keys:

        try:
            s = m.GetProp("_similarity")

            AllChem.Compute2DCoords(m)

            mols.append(m)

            inchi_keys.add(Chem.MolToInchiKey(m))

            similarity.append(float(s))

        except:

            None

inds = np.argsort(similarity)

mols = np.array(mols)[inds][0:100]

similarity = np.array(similarity)[inds][0:100]

mols = mols[0:100]

n_per_row = 10

scale = 300

n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

img = d2d.DrawMolecules(list(mols[0:100]), legends = [ str(round(s,3)) for s in similarity])

pic = open("/Users/alexanderhowarth/Desktop/NNs.svg", "w+")

d2d.FinishDrawing()

pic.write(str(d2d.GetDrawingText()))

pic.close()


