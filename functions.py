"""functionals for plotting embedding potential"""
import numpy as np
from pyscf.tools import cubegen
from pyscf.dft import gen_grid
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.pyscf_wrap_single import get_coulomb_repulsion,get_density_from_dm, get_dft_grid_stuff

#
#def cube_gen(mol,nx,ny,nz,margin=5):
#    """
#        nx : int
#            Number of grid point divisions in x direction.
#        ny : int
#            Number of grid point divisions in y direction.
#        nz : int
#            Number of grid point divisions in z direction."""
#    cc = cubegen.Cube(mol, nx, ny, nz,margin=5)
#    points = cc.get_coords()
#    return cc,points
    
def get_density(mol_A,mol_B,dmA,dmB,points):
    "return the electron density on a grid"
    rho0 = get_density_from_dm(mol_A, dmA, points)
    rho1 = get_density_from_dm(mol_B, dmB, points)
    rho_both = rho0 + rho1
    return rho0,rho1,rho_both
  

def get_elec(mol_tot,mol_A,mol_B,dmA,dmB,points):
    """
    Args:
        mol: molecular object
        dmA/dmB: 2D array
            density matrix, size of (ao,ao). 
            ao is the number of atomic orbitals
    """
    # Grid for plot
    rho0,rho1,rho_both = get_density(mol_A,mol_B,dmA,dmB,points)
    # Coulomb repulsion potential
    v_coul = get_coulomb_repulsion(mol_B,dmB,points)
    # Nuclear-electron attraction potential
    mol1_charges, mol1_coords = get_charges_and_coords(mol_B)
    v1_nuc0 = np.zeros(rho0.shape)
    for j, point in enumerate(points):
        for i in range(len(mol1_charges)):
            d = np.linalg.norm(point-mol1_coords[i])
            if d >= 1e-5:
                v1_nuc0[j] += - mol1_charges[i]/d 
    v_elec = v_coul + v1_nuc0
    return v_elec
    
def get_vT(rho0,rho1,rho_both,t_code='LDA_K_TF,'):
    """
    Args:
        xc_code: string
            see Pyscf instructions
        rho0/rho1: 1D array
            size of N (number of points)
        
    """
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho0, rho1)
    vt_emb = vts[0][0] - vts[1][0]- vts[2][0] 
    return vt_emb

def get_vXC(rho0,rho1,rho_both,xc_code='Slater,VWN5'):
    """
    Args:
        xc_code: string
            see Pyscf instructions
        rho0/rho1: 1D array
            size of N (number of points)
        
    """
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho0, rho1)
    vxc_emb = vxcs[0][0] - vxcs[1][0]-vxcs[2][0]
    return vxc_emb

def write_cc(vemb_tot,nx,ny,nz,outfile='emb.cube'):
    """    
    Args:
        vemb_tot: 1D array
            size of N
        mol_tot: mol object
    Kwargs:
        outfile: the file name

            """
    vemb_tot = vemb_tot.reshape(nx,ny,nz)
    outfile = 'emb.cube'
    # Write the potential
    cubegen.Cube.write(vemb_tot, outfile, 'Molecular embedding potential in real space')

def emb_potential(points,mola,dma,molb,dmb,xc_code='Slater,VWN5',t_code='LDA_K_TF',elec=True,non_k = True,non_xc = True):
    """Calculates the embedding potential and write out in
    cube format.

    """
    if elec == True:        
        v_elec = get_elec(system,mola,molb,dmA,dmB,points)
        print(v_elec)
        if non_k == True and non_xc == True:
            rho0,rho1,rho_both = get_density(mola,molb,dmA,dmB,points)
            print(rho_both)
            vxc_emb = get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = v_elec + vxc_emb + vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_full_pot.cube','Molecular embedding potential in real space')
           # functions.write_cc(vemb_tot,cc.nx,cc.ny,cc.nz,outfile = 'emb.cube') 
        elif non_k == True and non_xc == False:
            vt_emb = get_vT(rho0,rho1,rho_both,t_code)
            vemb_tot = v_elec + vt_emb
            #write_cc(vemb_tot,outfile = 'emb_k.cube') 
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

