import matplotlib.pyplot as plt
import numpy as np
one_d = np.arange(0.,5.,0.1)
basises = ['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']
for basis in basises:
    vemb = np.load(basis + '.npy')
#aug-ccpvdz = np.load('aug-cc-pvdz.npy')
#aug-ccpvtz = np.load('aug-cc-pvtz.npy')
#aug-ccpvqz = np.load('aug-cc-pvqz.npy')
    plt.plot(one_d,vemb,label = basis)
plt.legend()
plt.savefig('aug',dpi = 800)
plt.show()

