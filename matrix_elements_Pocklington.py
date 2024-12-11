# Import extension Better Comments for beauty
import better

#! This is a functions for calculating elemets of matrix equation 

#* Import libraries
import numpy as np #* Library for calculations 
from scipy import linalg #* Solver of matrix equations
import scipy.integrate as integrate #* Numerical integration

#* Define constants
c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12 #* Also define an operating frequency 

# Green function for Helmholtz equation
def Green_function(r_n, r_m, omega):
    rnm = np.linlag.norm(r_m-r_n, ord = 2)
    return 1j * omega * mu0/(4*np.pi)  * np.exp(-1j * omega/c * rnm)/rnm

# Derivatives for Green function
def derGreen_function(r_n, r_m, omega):
    rnm = np.linlag.norm(r_m-r_n, ord = 2)
    dz, k = r_m[2] - r_n[2], omega/c
    return 1j/(4*np.pi * omega * eps0) * dz (1 + 1j * k * rnm)  * np.exp(-1j * k * rnm)/(rnm ** 3)

def derderGreen_function(r_n, r_m, omega):
    rnm = np.linlag.norm(r_m-r_n, ord = 2)
    dz, k = r_m[2] - r_n[2], omega/c
    # Formula from Shuras ass
    poly_part = 2 * dz ** 2 + 3j * k * dz ** 2 * rnm - (1-k**2 * dz ** 2) * rnm ** 2 -\
        1j * k * rnm ** 3 + k ** 2 * rnm ** 4
    # Formula from Indian masters
    # poly_part = 3 * dz ** 2 + 3j * k * dz ** 2 * rnm - (1-k**2 * dz ** 2) * rnm ** 2 -\
    #     1j * k * rnm ** 3 + k ** 2 * rnm ** 4
    return poly_part * 1j * omega * mu0/(4 * np.pi * k ** 2 * rnm ** 5) * np.exp(-1j * k * rnm)

def calculate_impedance_Pocklington (R, element_num, element_radii, wavenumber, delta_z, frequency):

    def Greens_function (x, m, n, i, j):
        rmn = np.sqrt((x-R[n][j,2])**2 + (R[m][i,1] - R[n][j,1] - element_radii[0])**2 + (R[m][i,0]-R[n][j,0])**2)
        return (1j*frequency*mu0*np.exp(-1j*wavenumber*rmn)/(2*rmn))
    
    ZGreen_function = lambda z: Green_function(
        np.array([r_n[0], r_n[1], z]), 
        r_m, omega)
    
    def calculate_impedance_single (m, n, i, j): #* Using single quad
        return (integrate.quad(lambda x: (Greens_function(x, m, n, i, j).real), R[n][j,2] - delta_z, R[n][j,2] + delta_z)[0] + 1j * integrate.quad(lambda x: (Greens_function(x, m, n, i, j).imag), R[n][j,2] - delta_z, R[n][j,2] + delta_z)[0])
    
    def add_func (z1, m,n,i,j):
        rmn = np.sqrt((R[m][i,2]-z1)**2 + (R[m][i,1] - R[n][j,1] - element_radii[0])**2 + (R[m][i,0]-R[n][j,0])**2)
        return 1j/(8*np.pi*frequency*np.pi*eps0)* ((R[m][i,2]-z1) * (1+1j*wavenumber*rmn) * np.exp(-1j*wavenumber*rmn)/rmn**3)
    
    
    #* Compute block matrix of impedance
    impedance_block = []
    for m in range(0, len(element_radii)):
        impedance_row = []
        #* Filling the block using its teplitz structure
        for n in range (0, len(element_radii)):
            impedance_mn = np.zeros((len(R[m]), len(R[n])), dtype = complex)
            for i in range (len(R[m]) + len(R[n])):
                impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))] = calculate_impedance_single(m, n, max(0, len(R[m])-i-1), max(0, i-len(R[m]))) + add_func(R[n][max(0, i-len(R[m])),2]+delta_z,m, n, max(0, len(R[m])-i-1), max(0, i-len(R[m]))) - add_func(R[n][max(0, i-len(R[m])),2]-delta_z,m, n, max(0, len(R[m])-i-1), max(0, i-len(R[m])))
                for k in range (min( min(len(R[m]), len(R[n])), i+1, len(R[m]) + len(R[n]) - i)):
                    impedance_mn[max(0, len(R[m])-i-1) + k, max(0, i-len(R[m])) + k] = impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))]
            impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)
    return impedance_block

def calculate_voltage_Pocklington (R, source_position, driven_field, delta_z) :
    voltage = []
    for m in range (len(R)):
        voltage_row = np.zeros(len(R[m]))
        for i in range (len(R[m])):
            for k in range(len(source_position)):
                if all(source_position[k] == R[m][i,:]) :
                    voltage_row[i] = (driven_field / delta_z)
        voltage.append(voltage_row)
    return voltage