
import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm
c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_voltage (R_block, driven_voltage, source_position, delta_r) :
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    voltage_block = []
    for m in range (len(R_block)):
        voltage_row = np.zeros(len(R_block[m]))
        for i in range (len(R_block[m])):
            for k in range(len(source_position)):
                if all(source_position[k] == R_block[m][i,:]) :
                    voltage_row[i] = (driven_voltage)
        voltage_block.append(voltage_row)
        
    voltage = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        voltage[cum_n[i]:cum_n[i+1]] = voltage_block[i]
    return voltage_block, voltage

def ReGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.cos(- k * rmn) / rmn
def ImGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.sin(- k * rmn) / rmn

def RederderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real  
def ImderderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
def ImderderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).real
def ImderderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real
def ImderderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def RederderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).real
def ImderderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).imag

def RederderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).real
def ImderderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).imag


def Zmn_double (m, n, i, j, antenna, R_block, delta_r, radius, omega):

    a_m, a_n = radius, radius
    
    phi_m = np.atan2(antenna[m+1][1] - antenna[m][1], antenna[m+1][0] - antenna[m][0])
    phi_n = np.atan2(antenna[n+1][1] - antenna[n][1], antenna[n+1][0] - antenna[n][0])
    
    theta_m = np.atan2(antenna[m+1][2] - antenna[m][2], np.sqrt((antenna[m+1][0] - antenna[m][0])**2 + (antenna[m+1][1] - antenna[m][1])**2))
    theta_n = np.atan2(antenna[n+1][2] - antenna[n][2], np.sqrt((antenna[n+1][0] - antenna[n][0])**2 + (antenna[n+1][1] - antenna[n][1])**2))

    r_m = R_block[m][i] + np.array([-a_m*np.sin(theta_m)*np.cos(phi_m), -a_m*np.sin(theta_m)*np.sin(phi_m), a_m*np.cos(theta_m)])
    r_n = R_block[n][j]

    tau_m = (antenna[m+1] - antenna[m]) / np.linalg.norm(antenna[m+1] - antenna[m])
    tau_n = (antenna[n+1] - antenna[n]) / np.linalg.norm(antenna[n+1] - antenna[n])
    
    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    Z_0 = 1j*omega*mu0 / (4*np.pi) * np.dot(tau_m, tau_n) * delta_r**2 * (integrate.dblquad(ReGreen_function_double, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImGreen_function_double, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[0] <= 1e-9 :
        Z_x = 0
    else :
        Z_x = 1j/(4*np.pi * omega * eps0) * tau_m[0] * tau_n[0] * delta_r**2  * (integrate.dblquad(RederderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])

    if tau_n[1] * tau_m[1] <= 1e-9 :
        Z_y = 0
    else :
        Z_y = 1j/(4*np.pi * omega * eps0) * tau_m[1] * tau_n[1] * delta_r**2  * (integrate.dblquad(RederderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[2] * tau_m[2] <= 1e-9 :
        Z_z = 0
    else :
        Z_z = 1j/(4*np.pi * omega * eps0) * tau_m[2] * tau_n[2] * delta_r**2  * (integrate.dblquad(RederderZZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderZZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0] <= 1e-9 :
        Z_xy = 0
    else :
        Z_xy = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0] <= 1e-9 :
        Z_xz = 0
    else :
        Z_xz = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2] <= 1e-9 :
        Z_yz = 0
    else :
        Z_yz = 1j/(4*np.pi * omega * eps0) * (tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) * delta_r**2  * (integrate.dblquad(RederderYZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderYZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    return Z_0 + Z_x + Z_y + Z_z + Z_xy + Z_xz + Z_yz


def ReGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.cos(- k * rmn) / rmn
def ImGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.sin(- k * rmn) / rmn

def RederderXXGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real 
def ImderderXXGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
def ImderderYYGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderZZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).real
def ImderderZZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real
def ImderderXYGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def RederderXZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).real
def ImderderXZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).imag

def RederderYZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).real
def ImderderYZGreen_function_single(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).imag


def Zmn_single (m, n, i, j, antenna, R_block, delta_r, radius, omega):

    a_m, a_n = radius, radius
    
    phi_m = np.atan2(antenna[m+1][1] - antenna[m][1], antenna[m+1][0] - antenna[m][0])
    phi_n = np.atan2(antenna[n+1][1] - antenna[n][1], antenna[n+1][0] - antenna[n][0])
    
    theta_m = np.atan2(antenna[m+1][2] - antenna[m][2], np.sqrt((antenna[m+1][0] - antenna[m][0])**2 + (antenna[m+1][1] - antenna[m][1])**2))
    theta_n = np.atan2(antenna[n+1][2] - antenna[n][2], np.sqrt((antenna[n+1][0] - antenna[n][0])**2 + (antenna[n+1][1] - antenna[n][1])**2))

    r_m = R_block[m][i] + np.array([-a_m*np.sin(theta_m)*np.cos(phi_m), -a_m*np.sin(theta_m)*np.sin(phi_m), a_m*np.cos(theta_m)])
    r_n = R_block[n][j]

    tau_m = (antenna[m+1] - antenna[m]) / np.linalg.norm(antenna[m+1] - antenna[m])
    tau_n = (antenna[n+1] - antenna[n]) / np.linalg.norm(antenna[n+1] - antenna[n])
    
    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    Z_0 = 1j*omega*mu0 / (4*np.pi) * np.dot(tau_m, tau_n) * delta_r**2 * (integrate.quad(ReGreen_function_single, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImGreen_function_single, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[0] <= 1e-9 :
        Z_x = 0
    else :
        Z_x = 1j/(4*np.pi * omega * eps0) * tau_m[0] * tau_n[0] * delta_r**2  * (integrate.quad(RederderXXGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderXXGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])

    if tau_n[1] * tau_m[1] <= 1e-9 :
        Z_y = 0
    else :
        Z_y = 1j/(4*np.pi * omega * eps0) * tau_m[1] * tau_n[1] * delta_r**2  * (integrate.quad(RederderYYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderYYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[2] * tau_m[2] <= 1e-9 :
        Z_z = 0
    else :
        Z_z = 1j/(4*np.pi * omega * eps0) * tau_m[2] * tau_n[2] * delta_r**2  * (integrate.quad(RederderZZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderZZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0] <= 1e-9 :
        Z_xy = 0
    else :
        Z_xy = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) * delta_r**2  * (integrate.quad(RederderXYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderXYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0] <= 1e-9 :
        Z_xz = 0
    else :
        Z_xz = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) * delta_r**2  * (integrate.quad(RederderXZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderXZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    if tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2] <= 1e-9 :
        Z_yz = 0
    else :
        Z_yz = 1j/(4*np.pi * omega * eps0) * (tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) * delta_r**2  * (integrate.quad(RederderYZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.quad(ImderderYZGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    return Z_0 + Z_x + Z_y + Z_z + Z_xy + Z_xz + Z_yz


def Zmn (m, n, i, j, antenna, R_block, delta_r, radius, omega):
    
    a_m, a_n = radius, radius
    
    phi_m = np.atan2(antenna[m+1][1] - antenna[m][1], antenna[m+1][0] - antenna[m][0])
    phi_n = np.atan2(antenna[n+1][1] - antenna[n][1], antenna[n+1][0] - antenna[n][0])
    
    theta_m = np.atan2(antenna[m+1][2] - antenna[m][2], np.sqrt((antenna[m+1][0] - antenna[m][0])**2 + (antenna[m+1][1] - antenna[m][1])**2))
    theta_n = np.atan2(antenna[n+1][2] - antenna[n][2], np.sqrt((antenna[n+1][0] - antenna[n][0])**2 + (antenna[n+1][1] - antenna[n][1])**2))

    r_m = R_block[m][i] + np.array([-a_m*np.sin(theta_m)*np.cos(phi_m), -a_m*np.sin(theta_m)*np.sin(phi_m), a_m*np.cos(theta_m)])
    r_n = R_block[n][j]
    
    if np.linalg.norm(r_m - r_n) < 3e-2 :
        return Zmn_double(m, n, i, j, antenna, R_block, delta_r, radius, omega)
    else :
        return Zmn_single(m, n, i, j, antenna, R_block, delta_r, radius, omega)


def calculate_impedance (antenna, R_block, delta_r, radius, frequency):
    
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in  tqdm(range(0, len(R_block))):
        impedance_row = []
        for n in range (0, len(R_block)):
            impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
            tau_m = (antenna[m+1] - antenna[m]) / np.linalg.norm(antenna[m+1] - antenna[m])
            tau_n = (antenna[n+1] - antenna[n]) / np.linalg.norm(antenna[n+1] - antenna[n])
            if m == n or np.linalg.norm(np.cross(tau_m, tau_n)) <= 1e-9 :
                for i in range (len(R_block[m]) + len(R_block[n])):
                    impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn(m,n,max(0, len(R_block[m])-i-1),max(0, i-len(R_block[m])), antenna, R_block, delta_r, radius, 2*np.pi*frequency)
                    for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                        impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            else :
                for i in range(len(R_block[m])):
                    for j in range(len(R_block[n])):
                        impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn(m,n,max(0, len(R_block[m])-i-1),max(0, i-len(R_block[m])), antenna, R_block, delta_r, radius, 2*np.pi*frequency)
            impedance_row.append(impedance_mn)   
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance