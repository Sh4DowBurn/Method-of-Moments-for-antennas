
import better 

import numpy as np
import matplotlib.pyplot as plt
from visualization import calculate_positions
from matrix_elements_Pocklington import calculate_impedance_Pocklington
from matrix_elements_Pocklington import calculate_voltage_Pocklington

#* Define constants
light_speed, mu0, eps0= 299792458., 4*np.pi*1e-7, 8.854e-12

def directional_pattern (frequency, delta_z, incident_voltage, element_position, element_length, wire_radius, source_position):
    omega = 2 * np.pi * frequency
    
    #* Calculate some parametres of system
    wavelength, wavenumber = light_speed / frequency, omega / light_speed
    element_num, R_block, R = calculate_positions(element_length, element_position, frequency, delta_z)
    impedance = calculate_impedance_Pocklington(R_block, element_num, wire_radius, delta_z, omega)
    incident_field = calculate_voltage_Pocklington(R_block, element_num, source_position, incident_voltage, delta_z)
    current = np.linalg.solve(impedance, incident_field)
    
     #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    current_block = []
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        current_block.append(current[cum_n[i]:cum_n[i+1]])
    
    E = [0]*len(element_length)
    for m in range (len(element_length)):
        z0 = np.arange(element_num[m]) * delta_z - element_length[m]/2
        Ei = lambda phi : np.sum(current_block[m]*np.exp(1j*wavenumber*R_block[m][:,1]*np.cos(phi))*np.exp(-1j*R_block[m][:,2]*np.sin(phi)))
        phi = np.linspace(1e-6, 2*np.pi-1e-6, 1000)
        E[m] = np.array([Ei(phi_i) for phi_i in phi])*(np.exp(delta_z*wavenumber*np.sin(phi))-1)/(wavenumber*np.sin(phi))
    P_total = np.abs(np.sum(np.array(E), axis=0))
    P_total = P_total / np.max(P_total)
    plt.polar(phi, P_total, label = 'Dp')
    plt.title("Directional pattern")
    plt.legend()
    return R, R_block, element_num, incident_field, impedance, current, current_block, P_total, phi