#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:39:00 2020

@author: mingxue
"""
import numpy as np
import os
from pyscf.tools import cubegen
from qcelemental.models import Molecule
from pyscf import gto
from taco.methods.scf_pyscf import get_pyscf_molecule
from taco.embedding.pyscf_wrap import PyScfWrap
from embedding_potential_plot import functions
import CCJob as ccj

root = os.getcwd()
# =============================================================================
# feed the structure
zr_file = ccj.find_file(root,extension="zr")
frags = ccj.zr_frag(zr_file)
# =============================================================================
# Define Molecules with QCElemental
A = Molecule.from_data(data=frags["A"])
B = Molecule.from_data(data=frags["B"])
# Define arguments
basis = 'aug-cc-pvdz'
method = 'hf'
xc_code = 'Slater,VWN5'
t_code = 'LDA_K_TF,'
args0 = {"mol": A, "basis": basis, "method": method}
args1 = {"mol": B, "basis": basis, "method": method}
embs = {"mol": A, "basis": basis, "method": 'hf',
        "xc_code": xc_code, "t_code": t_code}
# Make a wrap using taco
wrap = PyScfWrap(args0, args1, embs)
# Make molecule in pyscf
A_mol = get_pyscf_molecule(A, basis)
B_mol = get_pyscf_molecule(B, basis)

# Create supersystem AB
newatom = '\n'.join([A_mol.atom, B_mol.atom])
system = gto.M(atom=newatom, basis=basis)
# option1--density matrix from the calculations Pyscf
#dmA = wrap.method0.get_density()
#dmB = wrap.method1.get_density()
#dmA = np.load("dm0_fromSCF.npy")
#dmB = np.load("dm1_fromSCF.npy") 
# =============================================================================
# define a grid for plotting
# =============================================================================
cc = cubegen.Cube(system,nx=20, ny=20, nz=20,margin=5)
points = cc.get_coords()
def emb_potential(points,xc_code,t_code,elec=True,non_k = True,non_xc = True):
    """Calculates the embedding potential and write out in
    cube format.

    """
    if elec == True:        
        v_elec = functions.get_elec(system,A_mol,B_mol,dmA,dmB,points)
        print(v_elec)
        if non_k == True and non_xc == True:
            rho0,rho1,rho_both = functions.get_density(wrap.mol0,wrap.mol1,dmA,dmB,points)
            print(rho_both)
            vxc_emb = functions.get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = functions.get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = v_elec + vxc_emb + vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb.cube','Molecular embedding potential in real space')
           # functions.write_cc(vemb_tot,cc.nx,cc.ny,cc.nz,outfile = 'emb.cube') 
        elif non_k == True and non_xc == False:
            vt_emb = functions.get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = v_elec + vt_emb
            functions.write_cc(vemb_tot,outfile = 'emb_k.cube') 
        elif non_k == False and non_xc == True:
            vxc_emb = functions.get_vXC(rho0,rho1,rho_both,xc_code)
            vemb_tot = v_elec + vt_emb
            functions.write_cc(vemb_tot,outfile = 'emb_xc.cube') 
        else:
            vemb_tot = v_elec
            functions.write_cc(vemb_tot,outfile = 'emb_elec.cube')
    if elec == False:
        rho0,rho1,rho_both = functions.get_density(system,A_mol,B_mol,dmA,dmB,points)
        if non_k == True and non_xc == True:
            vxc_emb = functions.get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = functions.get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = vxc_emb + vt_emb
            functions.write_cc(vemb_tot,outfile = 'emb.cube') 
        elif non_k == True and non_xc == False:
            vt_emb = functions.get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = vt_emb
            functions.write_cc(vemb_tot,outfile = 'emb_k.cube') 
        elif non_k == False and non_xc == True:
            vxc_emb = functions.get_vXC(rho0,rho1,rho_both,xc_code)
            vemb_tot = vt_emb
            functions.write_cc(vemb_tot,outfile = 'emb_xc.cube') 
        else:
            print("Define the embedding potentail")
#emb_potential(points=points,xc_code = xc_code,t_code = t_code,elec=True,non_k = True,non_xc = True)
