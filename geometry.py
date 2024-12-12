# Import extension Better Comments for beauty
import better 

#! This is a function for calculating geometry of antenna

#* import library for maths and arrays
import numpy as np

def calculate_positions (element_length, element_position, frequency, delta_z) :
    
    #* define constants
    light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12 
    
    #* Calculate some parametres of system
    wavelength, wavenumber = light_speed / frequency, 2 * np.pi * frequency / light_speed
    
    #* calculating number of sigments on each element
    element_num  = np.zeros(len(element_length), dtype = int)
    for i in range (len(element_length)):
        element_num[i] = int(element_length[i]/delta_z) if int(element_length[i]/delta_z)%2!=0 else int(element_length[i]/delta_z)+1
        
    #* Define list of positions, where R[m][i] - the position of i-th segment on m-th element
    R_block = [0] * len(element_num)
    for m in range (0, len(element_num)):
        R_block[m] = np.zeros((element_num[m], 3))
        for i in range (0, len(R_block[m])):
            R_block[m][i, 0] = 0
            R_block[m][i, 1] = element_position[m]
            R_block[m][i, 2] = -element_num[m]*delta_z / 2 + delta_z * (1/2 + i)
    
    #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    R = np.zeros((num_elements,3), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        R[cum_n[i]:cum_n[i+1]] = R_block[i]
    return element_num, R_block, R

