
import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 

c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

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

def calculate_field (antenna, R_block, driven_voltage, delta_r) :
    field_block = []
    for m in range (len(R_block)):
        field_row = np.zeros(len(R_block[m]))
        for i in range (len(R_block[m])):
            for k in range(len(antenna.source_position)):
                if all(antenna.source_position[k] == R_block[m][i,:]) :
                    field_row[i] = (driven_voltage / delta_r)
        field_block.append(field_row)
    
    element_num = np.array(len(R_block))
    incident_field = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        incident_field[cum_n[i]:cum_n[i+1]] = field_block[i]
    return incident_field

def calculate_impedance (antenna, R_block, delta_r, frequency):
    
    element_num = np.array(len(R_block))
    impedance_block = []
    
    for m in range(0, len(R_block)):
        
        impedance_row = []
        for n in range (0, len(R_block)):
            impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
            for i in range (len(R_block[m]) + len(R_block[n])):
                impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_doublequad(m,n,i,j,antenna) #! TO-DO
                
                for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                    impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)

    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance