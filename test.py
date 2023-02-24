from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

mols = [ standardize(m) for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/LL1.sdf") ]

print(mols)

print(mols[0].GetProp("ID"))