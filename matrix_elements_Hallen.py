# Import extension Better Comments for beauty
import better

#! This is a code for calculating elemets of matrix equation 

#* Import libraries
import numpy as np #* Library for calculations 
from scipy import linalg #* Solver of matrix equations
import scipy.integrate as integrate #* Numerical integration

def calculate_impedance_Hallen (R, element_num, element_radii, wavenumber, wavelength, epsilon) :
    
    #* Functions for numerical calculation of impedance
    def calculate_impedance_double (m, n, i, j): #* Using double quad
        return (integrate.dblquad(Greens_function_real, R[m][i,2] - delta_z, R[m][i,2] + delta_z, lambda x: R[n][j,2] - delta_z, lambda x: R[n][j,2] + delta_z, args=(m,n,i,j), epsabs=1e-12, epsrel=1e-12)[0] + 1j * integrate.dblquad(Greens_function_imag, R[m][i,2] - delta_z, R[m][i,2] + delta_z, lambda x: R[n][j,2] - delta_z, lambda x: R[n][j,2]+ delta_z, args = (m,n,i,j), epsabs=1e-12, epsrel=1e-12)[0])
    def calculate_impedance_signle (m, n): #* Using single quad
        return (integrate.quad(lambda x: (Greens_function(x, R[m, 2]).real), R[n, 2] - delta_z, R[n, 2] + delta_z)[0] + 1j * integrate.quad(lambda x: (Greens_function(x, R[m, 2]).imag), R[n, 2] - delta_z, R[n, 2] + delta_z)[0])

    #* Define Green's function for our differential operator (1+k^(-2)* d^2/dz^2)
    def Greens_function (zn, zm): #* For single quad
        rmn = np.sqrt((zm - zn) ** 2  + element_radii[0] ** 2)
        return np.exp(-1j*wavenumber*rmn)/(4*np.pi*rmn)
    def Greens_function_real (zm, zn, m, n, i, j): #* Real part for dblquad
        rmn = np.sqrt((zm-zn)**2 + (R[m][i,1] - R[n][j,1] - element_radii[0])**2 + (R[m][i,0] - R[n][j,0])**2)
        return (np.exp(-1j*wavenumber*rmn)/(4*np.pi*rmn)).real
    def Greens_function_imag (zm, zn, m, n, i, j): #* Imaginary part for dblquad
        rmn = np.sqrt((zm-zn)**2 + (R[m][i,1] - R[n][j,1] - element_radii[0])**2 + (R[m][i,0] - R[n][j,0])**2)
        return (np.exp(-1j*wavenumber*rmn)/(4*np.pi*rmn)).imag
    
    #* Define spatial resolution
    delta_z = wavelength / epsilon

    #* Compute block matrix of impedance
    impedance_block = []
    for m in range(0, len(element_radii)):
        impedance_row = []
        #* Filling the block using its teplitz structure
        for n in range (0, len(element_radii)):
            impedance_mn = np.zeros((len(R[m]), len(R[n])), dtype = complex)
            for i in range (len(R[m]) + len(R[n])):
                if max(0, len(R[m])-i-1) != max(0, i-len(R[m])) :
                    impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))] = calculate_impedance_double(m, n, max(0, len(R[m])-i-1), max(0, i-len(R[m])))
                else :
                    impedance_mn[[max(0, len(R[m])-i-1), max(0, i-len(R[m]))]] = 1/(4*np.pi) * np.log((np.sqrt(1 + 4 * (element_radii[0] * delta_z) ** 2) + 1)/(np.sqrt(1 + 4 * (element_radii[0] * delta_z) ** 2) - 1)) - 1j * wavenumber * delta_z / (4 * np.pi)
                for k in range (min( min(len(R[m]), len(R[n])), i+1, len(R[m]) + len(R[n]) - i)):
                    impedance_mn[max(0, len(R[m])-i-1) + k, max(0, i-len(R[m])) + k] = impedance_mn[max(0, len(R[m])-i-1), max(0, i-len(R[m]))]
            impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)
    return impedance_block

def calculate_solution_Hallen (R, element_num, wavenumber, source_position) :
    homogeneous_block = []
    nonhomogeneous_block = []
    for m in range (len(element_num)):
        s = np.zeros((len(R[m]), 2), dtype = complex)
        b = np.zeros(len(R[m]), dtype = complex)
        for i in range (len(R[m])):
            s[i,0] = np.exp(1j * wavenumber * R[m][i,2])
            s[i,1] = np.exp(-1j * wavenumber * R[m][i,2])
            if R[m][i,1] == source_position[0,1]:
                if R[m][i,2] >= 0:
                    b[i] = -1j/(2*wavenumber) * np.exp(1j * wavenumber * abs(R[m][i,2] - source_position[0,2]))
                else :
                    b[i] = -1j/(2*wavenumber) * np.exp(-1j * wavenumber * abs(R[m][i,2] - source_position[0,2]))
        homogeneous_block.append(s)
        nonhomogeneous_block.append(b)
    return homogeneous_block, nonhomogeneous_block