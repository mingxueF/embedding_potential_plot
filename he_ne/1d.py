import matplotlib.pyplot as plt
import numpy as np
one_d = np.arange(-2.,10.,0.1)
basises = ['aug-cc-pVDZ']#'aug-cc-pVDZ','aug-cc-pVTZ','aug-cc-pVQZ','aug-cc-pV5Z']
for basis in basises:
    vemb = np.load('rho_FnT_diff'+basis + '.npy')
    vemb_ref = np.load('rho_ref_diff'+basis + '.npy')
#aug-ccpvdz = np.load('aug-cc-pvdz.npy')
#aug-ccpvtz = np.load('aug-cc-pvtz.npy')
#aug-ccpvqz = np.load('aug-cc-pvqz.npy')
    plt.plot(one_d,vemb,label = "FnT")
    plt.plot(one_d,vemb_ref,label = 'ref')
    plt.legend()
    plt.xlabel(basis)
    plt.grid(linestyle= '--',linewidth='0.5')
    plt.savefig('1drho'+basis,dpi = 800)
    plt.show()

