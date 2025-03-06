
import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm
c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def ReGreen_function(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.cos(-1j * omega/c * rmn) / rmn

def ImGreen_function(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.sin(-1j * omega/c * rmn) / rmn

def RederderXGreen_function(t_n, r_m, r_n, dr_m, dr_n, omega):
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * pR_px / R**2
    polypart1 = -(1j * omega/c / R**2 - (1 + 1j * omega/c * R) * 1j * omega/c / R**2 - 2 * (1 + 1j * omega/c * R) / R**3)
    polypart2 = -(1 + 1j * omega/c * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * omega/c * R)).real
    
def ImderderXGreen_function(t_n, r_m, r_n, dr_m, dr_n, omega):
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * pR_px / R**2
    polypart1 = -(1j * omega/c / R**2 - (1 + 1j * omega/c * R) * 1j * omega/c / R**2 - 2 * (1 + 1j * omega/c * R) / R**3)
    polypart2 = -(1 + 1j * omega/c * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * omega/c * R)).imag

def RederderYGreen_function(t_n, r_m, r_n, dr_m, dr_n, omega):
    t_m, k = 1/2, omega/c 
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * pR_py / R**2
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
    
def ImderderYGreen_function(t_n, r_m, r_n, dr_m, dr_n, omega):
    t_m, k = 1/2, omega/c 
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * pR_py / R**2
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag



def Zmn (m, n, i, j, antenna, R_block, delta_r, omega):

    phi_m, phi_n = antenna.angle[m], antenna.angle[n]
    a_m, a_n = antenna.radius[m], antenna.radius[n]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]
    
    dr_m = delta_r * np.array([np.cos(phi_m), np.sin(phi_m)])
    dr_n = delta_r * np.array([np.cos(phi_n), np.sin(phi_n)])
    
    
    Z_dphi = delta_r ** 2 * 1j*omega*mu0 / (4*np.pi) * np.cos(phi_m - phi_n) * (integrate.dblquad(ReGreen_function, 0, 1, lambda z1: 0, lambda z2: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImGreen_function, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
    Z_xy = 0
    if np.sin(phi_n) <= 1e-9 or np.sin(phi_m) <= 1e-9 :
        Z_y = 0
        Z_x = delta_r ** 2 * 1j/(4*np.pi * omega * eps0) * np.cos(phi_n) * np.cos(phi_m) * (integrate.quad(RederderXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
    else :       
        Z_y = delta_r ** 2 * 1j/(4*np.pi * omega * eps0) * np.sin(phi_n) * np.sin(phi_m) * (integrate.quad(RederderYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
        Z_x = 0
    return Z_dphi + Z_x + Z_y + Z_xy

def calculate_field (antenna, R_block, driven_voltage, delta_r) :
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    field_block = []
    for m in range (len(R_block)):
        field_row = np.zeros(len(R_block[m]))
        for i in range (len(R_block[m])):
            for k in range(len(antenna.source_position)):
                if all(antenna.source_position[k] == R_block[m][i,:]) :
                    field_row[i] = (driven_voltage / delta_r)
        field_block.append(field_row)
        
    incident_field = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        incident_field[cum_n[i]:cum_n[i+1]] = field_block[i]
    return incident_field

def calculate_impedance (antenna, R_block, delta_r, frequency):
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    impedance_block = []
    for m in range(0, len(R_block)):
        impedance_row = []
        for n in range (0, len(R_block)):
            impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
            for i in tqdm(range (len(R_block[m]))):
                for j in range (len(R_block[n])):
                    impedance_mn[i][j] = Zmn(m,n,i,j,antenna,R_block,delta_r,2*np.pi*frequency)
                impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)
    impedance = np.zeros((sum(element_num),sum(element_num)), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance