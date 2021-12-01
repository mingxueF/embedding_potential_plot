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
def ndsd2_switch_factor(rhoB):
    """
    This formula is constructed artificially after reasoning with the NDSD2 potential.
    It motivates from the condition in the Lastra et. al. 2008 paper.
    Details can be found in the theoretical notes.

    Input:

    rhoB : np.array((6, N))
        Array with the density derivatives,
        density = rhoB[0]
        grad = rhoB[1:3] (x, y, z) derivatives
        laplacian = rhoB[4]


    Output: Real-valued switching constant between 0 and 1. """

    #Setting a zero mask for avoiding to small densities in rhoB (for wpot):
    zero_maskB=np.where(rhoB>1e-10)

    #Preallocate sfactor
    sfactor = np.zeros(rhoB.shape)

    #Formula for f^{NDSD2}(rho_B)=(1-exp(-rho_B))
    sfactor[zero_maskB] = (1-np.exp(-(rhoB[zero_maskB])))

    return sfactor
def compute_kinetic_weizsacker_modified(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    # Experimental functional derivative with the 1/8 factor:
    # A zero mask is added to exclude areas where rho=0
    zero_mask = np.where(abs(rho_devs[0] - 0.0) > 1e-10)[0]
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += 1.0/8.0*(np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - 1.0/8.0*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot
    
def get_density(mol_A,mol_B,dmA,dmB,points):
    "return the electron density on a grid"
    rho0 = get_density_from_dm(mol_A, dmA, points)
    rho1 = get_density_from_dm(mol_B, dmB, points)
    rho_both = rho0 + rho1
    return rho0,rho1,rho_both
  

def get_elec(mol_tot,mol_A,mol_B,dmA,dmB,rho0,rho1,rho_both,points):
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

def compute_ldacorr_pyscf(rho, xc_code=',VWN5'):
    """Correlation energy functional."""
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return (exc, vxc[0])
    
def get_vT(mola,dma,rho0,molb,dmb,rho1,rho_both,points,t_code='LDA_K_TF,',nadT='TF'):
    """
    Args:
        xc_code: string
            see Pyscf instructions
        rho0/rho1: 1D array
            size of N (number of points)
        
    """
    if nadT == 'TF':
        ets,vts = get_dft_grid_stuff(t_code,rho_both,rho0,rho1)
        vt_emb = vts[0][0] - vts[1][0]- vts[2][0] 
    if nadT == 'NDCS':
        rhoa_devs = get_density_from_dm(mola, dma, points, deriv=3, xctype='meta-GGA')
        rhob_devs = get_density_from_dm(molb, dmb, points, deriv=3, xctype='meta-GGA')
        rho_tot = rhoa_devs[0] + rhob_devs[0]
        etf_tot, vtf_tot = compute_ldacorr_pyscf(rho_tot, xc_code='LDA_K_TF')
        etf_0, vtf_0 = compute_ldacorr_pyscf(rhoa_devs[0], xc_code='LDA_K_TF')
        sfactor = ndsd2_switch_factor(rhob_devs[0])                #NDSD2 switching function
        wpot = compute_kinetic_weizsacker_modified(rhob_devs)   #Limit potential (gamma=1)
        vt_emb = vtf_tot - vtf_0 + sfactor * 1/8 * wpot            #NDSD potential
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

def emb_potential(system,mola,dmA,molb,dmB,xc_code='Slater,VWN5',t_code='LDA_K_TF',nadT='TF',elec=True,non_k = True,non_xc = True):
    """Calculates the embedding potential and write out in
    cube format.
    change the cube resolution accordingly..
    """
    from pyscf.tools import cubegen
    cc = cubegen.Cube(system, nx=80, ny=80, nz=80,margin=5)
    points = cc.get_coords()
    rho0,rho1,rho_both = get_density(mola,molb,dmA,dmB,points)
    if elec == True:        
        v_elec = get_elec(system,mola,molb,dmA,dmB,rho0,rho1,rho_both,points)
        print(v_elec)
        if non_k == True and non_xc == True:
            vxc_emb = get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = get_vT(mola,dmA,rho0,molb,dmB,rho1,rho_both,points,t_code,nadT)
            vemb_tot = v_elec + vxc_emb + vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_full_pot.cube','Molecular embedding potential in real space')
           # functions.write_cc(vemb_tot,cc.nx,cc.ny,cc.nz,outfile = 'emb.cube') 
        elif non_k == True and non_xc == False:
            vt_emb = get_vT(mola,dmA,rho0,molb,dmB,rho1,rho_both,points,t_code,nadT)
            vemb_tot = v_elec +  vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_K_pot.cube','Molecular embedding potential in real space')
        elif non_k == False and non_xc == True:
            vxc_emb = get_vXC(rho0,rho1,rho_both,xc_code)
            vemb_tot = v_elec + vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_xc_pot.cube','Molecular embedding potential in real space')
        else:
            vemb_tot = v_elec
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_elecOnly_pot.cube','Molecular embedding potential in real space')
    if elec == False:
        if non_k == True and non_xc == True:
            vxc_emb = get_vXC(rho0,rho1,rho_both,xc_code)
            vt_emb = get_vT(mola,dmA,rho0,molb,dmB,rho1,rho_both,points,t_code,nadT)
            vemb_tot = v_elec +  vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_nad_pot.cube','Molecular embedding potential in real space')
        elif non_k == True and non_xc == False:
            vt_emb = get_vT(mola,dmA,rho0,molb,dmB,rho1,rho_both,points,t_code,nadT)
            vemb_tot = v_elec +  vt_emb
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_T_only_pot.cube','Molecular embedding potential in real space')
        elif non_k == False and non_xc == True:
            vxc_emb = get_vXC(rho0,rho1,rho_both,xc_code)
            vemb_tot = vemb_tot.reshape(cc.nx,cc.ny,cc.nz)
            cc.write(vemb_tot,'emb_xc_only_pot.cube','Molecular embedding potential in real space')
        else:
            print("Define the embedding potentail")

