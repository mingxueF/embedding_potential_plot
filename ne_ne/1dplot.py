#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:39:00 2020

@author: mingxue
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pyscf.tools import cubegen
from qcelemental.models import Molecule
from taco.embedding.pyscf_wrap_single import get_density_from_dm
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
basis_str = 'aug-cc-pV5Z'
basispath = '/Users/mingxue/code/kernel_scripts/basis'
with open(os.path.join(basispath,basis_str+'.nwchem'), 'r') as bf:
    basis = bf.read()
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
naoA = A_mol.nao_nr
# Create supersystem AB
newatom = '\n'.join([A_mol.atom, B_mol.atom])
system = gto.M(atom=newatom, basis=basis)
# option1--density matrix from the calculations Pyscf
#dmA = wrap.method0.get_density()
#dmB = wrap.method1.get_density()
nao_mol0 = A_mol.nao_nr()
nao_mol1 = B_mol.nao_nr()
nao_tot = nao_mol0 + nao_mol1
print(nao_tot)
dmApath= os.path.join(root,"dm_A")
dmBpath= os.path.join(root,"dm_B")
dmABpath= os.path.join(root,"dm_AB")
dmA_mp2 = np.load(os.path.join(dmApath,"dm0_mp2"+basis_str+".npy"))
dmA_ref = np.load(os.path.join(dmApath,"dm0_ref"+basis_str+".npy"))# hf from FnT
dmB = np.load(os.path.join(dmApath,"dm1_ref"+ basis_str +".npy")) # hf from FnT
dmB_mp2 = np.load(os.path.join(dmBpath,'dm0_mp2'+basis_str+'.npy'))
dmA_iso = np.load(os.path.join(dmApath,'dm0_iso'+basis_str+'.npy')) # mp2 
dmB_iso = np.load(os.path.join(dmBpath,'dmB_iso'+basis_str+'.npy')) # mp2
dmAB_mp2 = np.load(os.path.join(dmABpath,basis_str+'mp2.npy'))
# =============================================================================
# define a grid for plotting
# =============================================================================
#cc = cubegen.Cube(system,nx=0, ny=0, nz=20,margin=5)
#points = cc.get_coords()
one_d = np.arange(-2.,10.,0.1)
points = np.array([[0.,0.,z] for z in one_d])
#to obtain a matrix A,B in gas MP2 density to be subtracted by supermolecular AB mp2
dm_both = np.zeros((nao_tot,nao_tot))
dm_both[:nao_mol0,:nao_mol0] = dmA_iso
dm_both[nao_mol0:,nao_mol0:] = dmB_iso
dm_FnT = np.zeros((nao_tot,nao_tot))
dm_FnT[:nao_mol0,:nao_mol0] = dmA_mp2
dm_FnT[nao_mol0:,nao_mol0:] = dmB_mp2
#--------------------------------------------------------
# get the dm difference of supermolecular AB mp2
dmref_diff = dmAB_mp2 - dm_both
rho_AB_diff = get_density_from_dm(system,dmref_diff,points)
np.save('rho_ref_diff'+basis_str,rho_AB_diff)
dmFnT_diff = dm_FnT-dm_both
rhoFnT_diff = get_density_from_dm(system,dmFnT_diff,points)
np.save('rho_FnT_diff'+basis_str,rhoFnT_diff)
#-----------------------------------------------------------------------------
def emb_potential(points,xc_code,t_code,elec=True,non_k = True,non_xc = True):
    """Calculates the embedding potential and write out in
    cube format.

    """
    if elec == True:        
        v_elec = functions.get_elec(system,A_mol,B_mol,dmA_ref,dmB,points)
        print(v_elec)
        if non_k == True and non_xc == True:
            rho0,rho1,rho_both = functions.get_density(wrap.mol0,wrap.mol1,dmA_ref,dmB,points)
            print(rho_both)
            vxc_emb = functions.get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = functions.get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = v_elec + vxc_emb + vt_emb
            np.save(basis_str,vemb_tot)
            plt.plot(one_d,vemb_tot,label='emb_potential')
            plt.show()
            plt.savefig(basis_str,dpi = 800)
            #vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            #cc.write(vemb_tot,'emb.cube','Molecular embedding potential in real space')
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
emb_potential(points=points,xc_code = xc_code,t_code = t_code,elec=True,non_k = True,non_xc = True)