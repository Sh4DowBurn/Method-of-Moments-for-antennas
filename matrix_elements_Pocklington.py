# Import extension Better Comments for beauty
import better
from tqdm import tqdm
#! This is a functions for calculating elemets of matrix equation 

#* Import libraries
import numpy as np #* Library for calculations 
from scipy import linalg #* Solver of matrix equations
import scipy.integrate as integrate #* Numerical integration

#* Define constants
c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12 #* Also define an operating frequency 

# Green function for Helmholtz equation
def Green_function(r_n, r_m, omega):
    rnm = np.linalg.norm(r_m-r_n, ord = 2)
    return 1j * omega * mu0/(4*np.pi)  * np.exp(-1j * omega/c * rnm)/rnm

# Derivatives for Green function, divided by k squared
def derGreen_function(r_n, r_m, omega):
    rnm = np.linalg.norm(r_m-r_n, ord = 2)
    dz, k = r_m[2] - r_n[2], omega/c
    return 1j/(4*np.pi * omega * eps0) * dz * (1 + 1j * k * rnm)  * np.exp(-1j * k * rnm)/(rnm ** 3)

def derderGreen_function(r_n, r_m, omega):
    rnm = np.linalg.norm(r_m-r_n, ord = 2)
    dz, k = r_m[2] - r_n[2], omega/c
    # Formula from Shuras ass
    poly_part = 2 * dz ** 2 + 3j * k * dz ** 2 * rnm - (1-k**2 * dz ** 2) * rnm ** 2 -\
        1j * k * rnm ** 3 + k ** 2 * rnm ** 4
    # Formula from Indian masters
    # poly_part = 3 * dz ** 2 + 3j * k * dz ** 2 * rnm - (1-k**2 * dz ** 2) * rnm ** 2 -\
    #     1j * k * rnm ** 3 + k ** 2 * rnm ** 4
    return poly_part * 1j * omega * mu0/(4 * np.pi * k ** 2 * rnm ** 5) * np.exp(-1j * k * rnm)

# Calculating impedance coefficient using single quad approximation
def Zmn_single(r_n, r_m, omega, dz, wire_radius):
    # Creating function to integrate along the wire using collacations on surface
    ReZGreen_function = lambda z: Green_function(
        np.array([r_n[0], r_n[1], z]), 
        r_m + np.array([0, wire_radius, 0]), omega).real
    ImZGreen_function = lambda z: Green_function(
        np.array([r_n[0], r_n[1], z]), 
        r_m + np.array([0, wire_radius, 0]), omega).imag
    ZderGreen_function = lambda z:derGreen_function(
        np.array([r_n[0], r_n[1], z]), 
        r_m + np.array([0, wire_radius, 0]), omega)
    
    # Calculate to part of matrix
    Z_1 = integrate.quad(ReZGreen_function, r_n[2]-dz/2, r_n[2]+dz/2)[0] + 1j * integrate.quad(ImZGreen_function, r_n[2]-dz/2, r_n[2]+dz/2)[0]
    Z_2 = ZderGreen_function(r_n[2] + dz/2) - ZderGreen_function(r_n[2] - dz/2)
    return Z_1 + Z_2
            
            
def calculate_impedance_Pocklington (R, element_num, wire_radius, delta_z, omega):
    
    #* Compute block matrix of impedance
    impedance_block = []
    for m in range(0, len(R)):
        impedance_row = []
        #* Filling the block using its teplitz structure
        for n in range (0, len(R)):
            impedance_mn = np.zeros((len(R[m]), len(R[n])), dtype = complex)
            for i in range (len(R[m]) + len(R[n])):
                impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))] = Zmn_single(R[n][max(0, i-len(R[m]))], R[m][max(0, len(R[m])-i-1)], omega, delta_z, wire_radius)
                for k in range (min( min(len(R[m]), len(R[n])), i+1, len(R[m]) + len(R[n]) - i)):
                    impedance_mn[max(0, len(R[m])-i-1) + k, max(0, i-len(R[m])) + k] = impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))]
            impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)
        
    #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance

def calculate_voltage_Pocklington (R, element_num, source_position, driven_field, delta_z) :
    voltage_block = []
    for m in range (len(R)):
        voltage_row = np.zeros(len(R[m]))
        for i in range (len(R[m])):
            for k in range(len(source_position)):
                if all(source_position[k] == R[m][i,:]) :
                    voltage_row[i] = (driven_field / delta_z)
        voltage_block.append(voltage_row)
    #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    voltage = np.zeros((num_elements), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        voltage[cum_n[i]:cum_n[i+1]] = voltage_block[i]
    return voltage_block, voltage



# Calculate and return impedance matrix
#def Z_matrix(R, wire_radius, dz, omega):
#    # Create Data-dict to avoid recalculations
#    Data = {}
#    Z = np.zeros((len(R), len(R)), dtype = complex)
#    for n, r_n in tqdm(enumerate(R)):
#        for m, r_m in enumerate(R):
#            delta_r = r_m - r_n
#            id_positive, id_negative = (delta_r[0], delta_r[1], delta_r[2]), (-delta_r[0], -delta_r[1], -delta_r[2])
#            if id_positive in Data or id_negative in Data:
#                Z[n][m] = Data[id_positive]
#            else:
#                Z[n][m] = Zmn_single(r_n, r_m, omega, dz, wire_radius)
#                Data[id_positive],Data[id_negative] = Z[n][m], Z[n][m]         
#    return Z