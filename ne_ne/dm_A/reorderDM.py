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
root = os.getcwd()
zr_file = ccj.find_file(root,extension="zr")
frags = ccj.zr_frag(zr_file)
basis_strs = ['aug-cc-pVDZ','aug-cc-pVTZ','aug-cc-pVQZ','aug-cc-pV5Z']
for basis_str in basis_strs:
    path = os.path.join(home,'wi_group/ne_ne',basis_str)
    dm_file = os.path.join(path,'A_MP2','Densmat_MP.txt')
    # =============================================================================
    #load DM
    dm = np.loadtxt(dm_file)
    mid0 = len(dm)//2
    dm = dm[:mid0]
    # =============================================================================
    # =============================================================================
    # Define Molecules with QCElemental
    with open(os.path.join(basisfiles, basis_str+'.nwchem'), 'r') as bf:
        ibasis = bf.read()    
    mol0 = gto.M(atom=frags['A'], basis=ibasis)
    nao_mol0 = mol0.nao_nr()
    print("AOs tot = ", nao_mol0)
    # Reshape the density matrix
    dm = 2.0*dm.reshape((nao_mol0, nao_mol0))
    # Order from qchem to pyscf
    atoms0 = []
    for i in range(mol0.natm):
        atoms0.append(int(mol0._atm[i][0])) #check mol._atm
    atoms0 = np.array(atoms0, dtype=int)
    print(atoms0)
    dm_ref = reorder_matrix(dm, 'qchem', 'pyscf', basis_str, atoms0)
    np.save('dm0_iso'+basis_str,dm_ref)
