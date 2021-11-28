#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:36:24 2020

@author: mingxue
"""
import os
import numpy as np
import pandas as pd

from pyscf import gto, scf, dft
from pyscf import lib
import CCJob as ccj
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat, NumInt
from pyscf.dft import gen_grid, libxc
from taco.embedding.pyscf_wrap_single import get_density_from_dm
from taco.translate.tools import reorder_matrix

home = '/Users/mingxue/code/kernel_scripts'
basisfiles = home+'/basis/'
basis_str = 'aug-cc-pV5Z'
root = os.getcwd()
zr_file = ccj.find_file(root,extension="zr")
frags = ccj.zr_frag(zr_file)
dm_file = 'AB_MP.txt'
# =============================================================================
#load DM
dm = np.loadtxt(os.path.join(root, dm_file))
mid0 = len(dm)//2
dm = dm[:mid0]
# =============================================================================
# =============================================================================
# Define Molecules with QCElemental
with open(os.path.join(basisfiles, basis_str+'.nwchem'), 'r') as bf:
    ibasis = bf.read()    
mol0 = gto.M(atom=frags['A'], basis=ibasis)
mol1 = gto.M(atom=frags['B'],basis= ibasis)
nao_mol0 = mol0.nao_nr()
nao_mol1 = mol1.nao_nr()
print("AOs tot = ", nao_mol0)
# Reshape the density matrix
dm = 2.0*dm.reshape((nao_mol0+nao_mol1, nao_mol0+nao_mol1))
# Order from qchem to pyscf
atoms = []
for i in range(mol0.natm):
    atoms.append(int(mol0._atm[i][0])) #check mol._atm
print(atoms)
for j in range(mol1.natm):
    atoms.append(int(mol1._atm[j][0]))
atoms = np.array(atoms, dtype=int)
print(atoms)
dm_ref = reorder_matrix(dm, 'qchem', 'pyscf', basis_str, atoms)
np.save('dmAB_mp2',dm)
