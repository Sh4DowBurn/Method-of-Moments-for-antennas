import numpy as np
import matplotlib.pyplot as plt
def calculate_dp (R, current, element_length, element_num, delta_z, wavenumber):
     #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    current_block = []
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        current_block.append(current[cum_n[i]:cum_n[i+1]])
    E = [0]*len(element_length)
    for m in range (len(element_length)):
        z0 = np.arange(element_num[m]) * delta_z - element_length[m]/2
        Ei = lambda phi : np.sum(current_block[m]*np.exp(1j*wavenumber*R[m][:,1]*np.cos(phi))*np.exp(-1j*R[m][:,2]*np.sin(phi)))
        phi = np.linspace(1e-6, 2*np.pi-1e-6, 1000)
        E[m] = np.array([Ei(phi_i) for phi_i in phi])*(np.exp(delta_z*wavenumber*np.sin(phi))-1)/(wavenumber*np.sin(phi))
    P_total = np.abs(np.sum(np.array(E), axis=0))
    P_total = P_total / np.max(P_total)
    plt.polar(phi, P_total, label = 'Dp')
    plt.title("Directional pattern")
    plt.legend()
    return P_total