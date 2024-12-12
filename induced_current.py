
import better 

import numpy as np
import matplotlib.pyplot as plt
from geometry import calculate_positions
from matrix_elements_Pocklington import calculate_impedance_Pocklington
from matrix_elements_Pocklington import calculate_voltage_Pocklington
from visualization import plot_2dmodel

#* Define constants
light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_current (frequency, delta_z, incident_voltage, element_position, element_length, wire_radius, source_position):
    omega = 2 * np.pi * frequency
    
    #* Calculate some parametres of system
    wavelength, wavenumber = light_speed / frequency, omega / light_speed
    element_num, R_block, R = calculate_positions(element_length, element_position, frequency, delta_z)
    
    plot_2dmodel(R_block,source_position, sum(element_num), delta_z)
    
    impedance = calculate_impedance_Pocklington(R_block, element_num, wire_radius, delta_z, omega)
    incident_field = calculate_voltage_Pocklington(R_block, element_num, source_position, incident_voltage, delta_z)
    current = np.linalg.solve(impedance, incident_field)
    
    
    #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    element_currents = []
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        element_currents.append(current[cum_n[i]:cum_n[i+1]])
    
    return element_currents, current