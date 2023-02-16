# Molecular conformer generator
#
# Example:
#
# genConf.py -i file_input.sdf -o file_output.sdf
# -n number_of_conformers (optional, if not specified is based
# on the nomber of rotable bonds) -rms rms_threshold
# -e energy_window (optional, Kcal/mol) -t number_of_threads (if not specify 1)
# -ETKDG (optional, use the useExpTorsionAnglePrefs and useBasicKnowledge options)
# -logfile (Default: False, write a log file with info about the conformer generation)
# ----------------------------------------------------------
# ----------------------------------------------------------
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from concurrent import futures

import argparse
import copy
import time
import gzip

def genConf(m):

    ###########
    nmol = 1
    rms = -1
    rmspost = 0.2
    irmsd = "Not"
    threads = 1
    efilter = "Y"
    iener = "Not"
    nc = "X"
    inc = "Rotatable bond based"
    etkdg = False

    molName = ""

    mini=None
    info = ""
    nr = int(AllChem.CalcNumRotatableBonds(m))
    m4 = copy.deepcopy(m)
    m = Chem.AddHs(m)
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    doublestereolist=[]
    for b in range(0, m.GetNumBonds()):
        st = str(m.GetBondWithIdx(b).GetStereo())
        ty = str(m.GetBondWithIdx(b).GetBondType())
        if ty == "DOUBLE" and (st == 'STEREOZ' or st == "STEREOCIS"):
                doublestereolist.append((b, "STEREO Z"))
        elif ty == "DOUBLE" and (st == "STEREOE" or st == "STEREOTRANS"):
                doublestereolist.append((b, "STEREO E"))
    Chem.AssignStereochemistry(m, cleanIt=True, flagPossibleStereoCenters=True)
    chiralcenter = Chem.FindMolChiralCenters(m)+doublestereolist
    if nc == "X":
        if nr < 3:
            nc = 50
        elif nr > 6:
            nc = 300
        else:
            nc = nr**3

    m3 = copy.deepcopy(m)
    m5 = copy.deepcopy(m)
    ids=AllChem.EmbedMultipleConfs(m, numConfs=nc, pruneRmsThresh=rms, randomSeed=1, useExpTorsionAnglePrefs=etkdg, useBasicKnowledge=etkdg)
    if rms == -1 and efilter == "Y":
        if len(ids) != nc:
            info = "WARNING: " + molName + " generated less molecules than those required\n"
    numconf = m.GetNumConformers()

    if numconf == 0:
        m = copy.deepcopy(m3)
        ids=AllChem.EmbedMultipleConfs(m, numConfs=nc, pruneRmsThresh=rms, randomSeed=1)
        info = "WARNING: Molecule number " + str(nmol) + " embed without ETKDG method, molecule name: " + molName + "\n"
    m2 = copy.deepcopy(m)
    diz = []
    diz2 = []
    diz3 = []

    if m.GetNumConformers() == 0:
        info = "ERROR: Impossible to generate conformers of molecule " + str(nmol) + ", molecule name: " + molName + "\n"
        o = m4
        o = Chem.AddHs(o)
        embd = AllChem.EmbedMolecule(o, randomSeed=1)
        if embd == -1:
            info = "ERROR: Impossible to generate conformers of molecule " + str(nmol) + ", molecule name: " + molName + "\n"
            o = m4
        diz3 = [(None, -1)]

    else:
        for id in ids:
            molec = m.GetConformer(id).GetOwningMol()
            doublestereolist=[]
            for b in range(0, molec.GetNumBonds()):
                st = str(molec.GetBondWithIdx(b).GetStereo())
                ty = str(molec.GetBondWithIdx(b).GetBondType())
                if ty == "DOUBLE" and (st == 'STEREOZ' or st == "STEREOCIS"):
                    doublestereolist.append((b, "STEREO Z"))
                elif ty == "DOUBLE" and (st == "STEREOE" or st == "STEREOTRANS"):
                    doublestereolist.append((b, "STEREO E"))
            Chem.AssignStereochemistry(molec, cleanIt=True, flagPossibleStereoCenters=True)
            confchiralcenter = Chem.FindMolChiralCenters(molec)+doublestereolist
            if confchiralcenter != chiralcenter:
                m.RemoveConformer(id)
        if m.GetNumConformers() == 0:
            m = m5
            ids=AllChem.EmbedMultipleConfs(m, numConfs=nc, pruneRmsThresh=rms, randomSeed=1)
            for id in ids:
                molec = m.GetConformer(id).GetOwningMol()
                for b in range(0, molec.GetNumBonds()):
                    st = str(molec.GetBondWithIdx(b).GetStereo())
                    ty = str(molec.GetBondWithIdx(b).GetBondType())
                    if ty == "DOUBLE" and (st == 'STEREOZ' or st == "STEREOCIS"):
                        doublestereolist.append((b, "STEREO Z"))
                    elif ty == "DOUBLE" and (st == "STEREOE" or st == "STEREOTRANS"):
                        doublestereolist.append((b, "STEREO E"))
                Chem.AssignStereochemistry(molec, cleanIt=True, flagPossibleStereoCenters=True)
                confchiralcenter = Chem.FindMolChiralCenters(molec)+doublestereolist
                if confchiralcenter != chiralcenter:
                    if info != "":
                        info = info + "\n"
                    info = "WARNING: one or more conformer of Molecule number " + str(nmol) + " were excluded becouse it/they has/have different isomerism respect the input: " + molName + "\n"
                    m.RemoveConformer(id)
        try:
            if AllChem.MMFFHasAllMoleculeParams(m) == True:
                sm = copy.deepcopy(m)
                try:
                    for id in ids:
                        prop = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
                        ff = AllChem.MMFFGetMoleculeForceField(m, prop, confId=id)
                        ff.Minimize()
                        en = float(ff.CalcEnergy())
                        econf = (en, id)
                        diz.append(econf)
                except:
                    m = sm
                    for id in ids:
                        ff = AllChem.UFFGetMoleculeForceField(m, confId=id)
                        ff.Minimize()
                        en = float(ff.CalcEnergy())
                        econf = (en, id)
                        diz.append(econf)
                    if info != "":
                        info = info + "WARNING: Molecule number " + str(nmol) + " optimized with UFF force field, molecule name: " + molName + "\n"
                    else:
                        info = "WARNING: Molecule number " + str(nmol) + " optimized with UFF force field, molecule name: " + molName + "\n"
            else:
                for id in ids:
                    ff = AllChem.UFFGetMoleculeForceField(m, confId=id)
                    ff.Minimize()
                    en = float(ff.CalcEnergy())
                    econf = (en, id)
                    diz.append(econf)
                if info != "":
                    info = info + "WARNING: Molecule number " + str(nmol) + " optimized with UFF force field, molecule name: " + molName + "\n"
                else:
                    info = "WARNING: Molecule number " + str(nmol) + " optimized with UFF force field, molecule name: " + molName + "\n"
        except:
            m = m2
            if info != "":
                info = info + "ERROR: Unable to minimize molecule number: " + str(nmol)+ ", molecule name: " + molName + "\n"
            else:
                info = "ERROR: Unable to minimize molecule number: " + str(nmol)+ ", molecule name: " + molName + "\n"
            for id in ids:
                en = None
                econf = (en, id)
                diz.append(econf)

        if efilter != "Y":
            n, diz2, mini = ecleaning(m, diz, efilter)
        else:
            n = m
            diz2 = copy.deepcopy(diz)
            diz.sort()
            mini = float(diz[0][0])

        if rmspost != False and n.GetNumConformers() > 1:
            o, confidlist,enval = postrmsd(n, diz2, rmspost)
        else:
            o = n
            diz3 = diz2

    return o, confidlist,enval

def ecleaning(m, diz, efilter):
    diz.sort()
    mini = float(diz[0][0])
    sup = mini + efilter
    n = Chem.Mol(m)
    n.RemoveAllConformers()
    n.AddConformer(m.GetConformer(int(diz[0][1])))
    diz2=[[float(diz[0][0]), int(diz[0][1])]]
    del diz[0]
    for x,y in diz:
        if x <= sup:
            n.AddConformer(m.GetConformer(int(y)))
            uni = [float(x), int(y)]
            diz2.append(uni)
        else:
            break
    return n, diz2, mini

def postrmsd(n, diz2, rmspost):
    diz2.sort()
    o = Chem.Mol(n)
    o.RemoveAllConformers()
    confidlist = [diz2[0][1]]
    enval = [diz2[0][0]]
    nh = Chem.RemoveHs(n)
    nh = Chem.DeleteSubstructs(nh, Chem.MolFromSmiles('F'))
    nh = Chem.DeleteSubstructs(nh, Chem.MolFromSmiles('Br'))
    nh = Chem.DeleteSubstructs(nh, Chem.MolFromSmiles('Cl'))
    nh = Chem.DeleteSubstructs(nh, Chem.MolFromSmiles('I'))
    del diz2[0]
    for z,w in diz2:
        confid = int(w)
        p=0
        for conf2id in confidlist:
            rmsd = AllChem.GetBestRMS(nh, nh, prbId=confid, refId=conf2id)
            if rmsd < rmspost:
                p=p+1
                break
        if p == 0:
            confidlist.append(int(confid))
            enval.append(float(z))
    for id in confidlist:
        o.AddConformer(n.GetConformer(id))

    return o, confidlist,enval
