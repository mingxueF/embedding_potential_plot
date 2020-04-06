"""Compute the embedding potential on a grid and export it as a cube file."""
import numpy as np
import os
from qcelemental.models import Molecule
from pyscf import gto
from pyscf.tools import cubegen
from pyscf.dft import gen_grid
from taco.methods.scf_pyscf import get_pyscf_molecule
from taco.embedding.pyscf_wrap import PyScfWrap, get_charges_and_coords
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_dft_grid_stuff
from taco.embedding.cc_gridfns import coulomb_potential_grid
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
basis = 'cc-pvqz'
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
dmA = np.load("dm0_fromSCF.npy")
dmB = np.load("dm1_fromSCF.npy")    
# option2--import the density matric from outside
#basis_A = A_mol.nao_nr()     # the number of AO
#basis_B = B_mol.nao_nr()
#basis_AB = basis_A + basis_B 
#dmA_fde_mp2 = np.loadtxt("/home/fu/code/taco_public/examples/embedding/glycine_h2o/3.154/FDE_State0_tot_dens.txt",\
#                 max_rows=basis_A*basis_A).reshape(basis_A,basis_A)
#dmB_fde_mp2 = np.loadtxt("/home/fu/code/taco_public/examples/embedding/glycine_h2o/3.154/State_mp2_gs_B.txt",\
#                 max_rows=basis_B*basis_B).reshape(basis_B,basis_B)
#dmAB_mp2 = np.loadtxt("/home/fu/code/taco_public/examples/embedding/glycine_h2o/3.154/Densmat_MP.txt",\
#                 max_rows=basis_AB*basis_AB).reshape(basis_AB,basis_AB)
def emb_potential(mol, nx=80, ny=80, nz=80,option = 1,outfile="emb.cube"):
    """Calculates the embedding potential and write out in
    cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        option: int
             1: the complete embedding potential(default)
             0: only electrostatic part
    """
    cc = cubegen.Cube(mol, nx, ny, nz,margin=5)

    points = cc.get_coords()
    grids = gen_grid.Grids(system)
    grids.level = 4
    grids.build()
    # Grid for plot
    rho0 = get_density_from_dm(wrap.mol0, dmA, points)
    rho1 = get_density_from_dm(wrap.mol1, dmB, points)
    rho_both = rho0 + rho1
    rho1_grid = get_density_from_dm(wrap.mol1, dmB, grids.coords)
    # write the electron density to a cube file    
    rho = rho_both.reshape(cc.nx,cc.ny,cc.nz)
    cc.write(rho,'rho.cube', comment='Electron density in real space (e/Bohr^3)')
    # Coulomb repulsion potential
    v_coul = coulomb_potential_grid(points, grids.coords, grids.weights, rho1_grid)
    # Nuclear-electron attraction potential
    mol1_charges, mol1_coords = get_charges_and_coords(B_mol)
    v1_nuc0 = np.zeros(rho0.shape)
    for j, point in enumerate(points):
        for i in range(len(mol1_charges)):
            d = np.linalg.norm(point-mol1_coords[i])
            if d >= 1e-5:
                v1_nuc0[j] += - mol1_charges[i]/d 
    # DFT nad potential
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho0, rho1)
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho0, rho1)
    vxc_emb = vxcs[0][0] - vxcs[1][0]-vxcs[2][0]
    vt_emb = vts[0][0] - vts[1][0]- vts[2][0] 
    if option == 1:
        vemb_tot = v_coul + v1_nuc0 + vxc_emb + vt_emb
        vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
        outfile = 'emb.cube'
        # Write the potential
        cc.write(vemb_tot, outfile, 'Molecular embedding electrostatic potential in real space')
    else:
        vemb_tot = v_coul + v1_nuc0
        vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
        outfile = 'emb_elec.cube'
        # Write the potential
        cc.write(vemb_tot, outfile, 'Molecular embedding electrostatic potential in real space')
    return vemb_tot
emb_potential(mol=system,nx=20,ny=20,nz=20)
