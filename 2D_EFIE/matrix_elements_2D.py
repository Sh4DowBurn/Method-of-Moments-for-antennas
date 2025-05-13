
import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm
c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

epsabs, epsrel = 1e-6, 1e-6

def calculate_voltage (source_position, R_block, driven_voltage, delta_r) :
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    field_block = []
    for m in range (len(R_block)):
        field_row = np.zeros(len(R_block[m]))
        for i in range (len(R_block[m])):
            for k in range(len(source_position)):
                if np.linalg.norm(source_position[k] - R_block[m][i,:]) <= delta_r/2 :
                    field_row[i] = (driven_voltage)
        field_block.append(field_row)
        
    incident_field = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        incident_field[cum_n[i]:cum_n[i+1]] = field_block[i]
    return incident_field

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

def Zmn_double (m, n, i, j, angles, radii, R_block, delta_r, omega):

    phi_m, phi_n = angles[m], angles[n]
    a_m, a_n = radii[m], radii[n]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]
    
    dr_m = delta_r * np.array([np.cos(phi_m), np.sin(phi_m)])
    dr_n = delta_r * np.array([np.cos(phi_n), np.sin(phi_n)])
    
    
    Z_dphi = 1j*omega*mu0 / (4*np.pi) * np.cos(phi_m - phi_n) * delta_r**2 * (integrate.dblquad(ReGreen_function_double, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.dblquad(ImGreen_function_double, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])
    
    if np.sin(phi_n + phi_m) <= 1e-9 :
        Z_xy = 0
    else :
        Z_xy = 1j/(4*np.pi * omega * eps0) * np.sin(phi_m + phi_n) * delta_r**2 * (integrate.dblquad(RederderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.dblquad(ImderderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])
    
    if np.sin(phi_n) <= 1e-9 or np.sin(phi_m) <= 1e-9 :
        Z_y = 0
    else :       
        Z_y = 1j/(4*np.pi * omega * eps0) * np.sin(phi_n) * np.sin(phi_m) * delta_r**2 * (integrate.dblquad(RederderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.dblquad(ImderderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])

    if np.cos(phi_n) <= 1e-9 or np.cos(phi_m) <= 1e-9 :
        Z_x = 0
    else :
        Z_x = 1j/(4*np.pi * omega * eps0) * np.cos(phi_n) * np.cos(phi_m) * delta_r**2  * (integrate.dblquad(RederderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.dblquad(ImderderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])
    
    return Z_dphi+Z_x+Z_y+Z_xy

def calculate_impedance_double (R_block, angles, radii, delta_r, frequency):
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in range(0, len(R_block)):
        impedance_row = []
        for n in range (0, len(R_block)):
            if m == n or np.abs(angles[n]-angles[m]) <= 1e-9 :
                impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
                for i in range (len(R_block[m]) + len(R_block[n])):
                    impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_double(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
                    for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                        impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            else :
                for i in range(len(R_block[m])):
                    for j in range(len(R_block[n])):
                        impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_double(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
            impedance_row.append(impedance_mn)   
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance


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

def Zmn_single (m, n, i, j, angles, radii, R_block, delta_r, omega):

    phi_m, phi_n = angles[m], angles[n]
    a_m, a_n = radii[m], radii[n]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]
    
    dr_m = delta_r * np.array([np.cos(phi_m), np.sin(phi_m)])
    dr_n = delta_r * np.array([np.cos(phi_n), np.sin(phi_n)])
    
    
    Z_dphi = 1j*omega*mu0 / (4*np.pi) * np.cos(phi_m - phi_n) * delta_r**2 * (integrate.quad(ReGreen_function_single, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.quad(ImGreen_function_single, 0, 1, args=(r_m, r_n, dr_m, dr_n, omega), epsabs=1e-6, epsrel=1e-6)[0])
    
    if np.sin(phi_n + phi_m) <= 1e-9 :
        Z_xy = 0
    else :
        Z_xy = 1j/(4*np.pi * omega * eps0) * np.sin(phi_m + phi_n) * delta_r**2 * (integrate.quad(RederderXYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.quad(ImderderXYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])
    
    if np.sin(phi_n) <= 1e-9 or np.sin(phi_m) <= 1e-9 :
        Z_y = 0
    else :       
        Z_y = 1j/(4*np.pi * omega * eps0) * np.sin(phi_n) * np.sin(phi_m) * delta_r**2 * (integrate.quad(RederderYYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.quad(ImderderYYGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])

    if np.cos(phi_n) <= 1e-9 or np.cos(phi_m) <= 1e-9 :
        Z_x = 0
    else :
        Z_x = 1j/(4*np.pi * omega * eps0) * np.cos(phi_n) * np.cos(phi_m) * delta_r**2  * (integrate.quad(RederderXXGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0] + 1j * integrate.quad(ImderderXXGreen_function_single, 0, 1, args = (r_m, r_n, dr_m, dr_n, omega), epsabs=epsabs, epsrel=epsrel)[0])
    
    return Z_dphi+Z_x+Z_y+Z_xy

def calculate_impedance_single (R_block, angles, radii, delta_r, frequency):
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in range(0, len(R_block)):
        impedance_row = []
        for n in range (0, len(R_block)):
            if m == n or np.abs(angles[n]-angles[m]) <= 1e-9 :
                impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
                for i in range (len(R_block[m]) + len(R_block[n])):
                    impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_single(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
                    for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                        impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            else :
                for i in range(len(R_block[m])):
                    for j in range(len(R_block[n])):
                        impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_single(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
            impedance_row.append(impedance_mn)   
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance



def ReGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.cos(- k * rmn) / rmn

def ImGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return np.sin(- k * rmn) / rmn

def RederderXXGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real
    
def ImderderXXGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
    
def ImderderYYGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real

def ImderderXYGreen_function_noquad(t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    t_m = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    return ((polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def Zmn_noquad (m, n, i, j, angles, radii, R_block, delta_r, omega):

    phi_m, phi_n = angles[m], angles[n]
    a_m, a_n = radii[m], radii[n]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]
    
    dr_m = delta_r * np.array([np.cos(phi_m), np.sin(phi_m)])
    dr_n = delta_r * np.array([np.cos(phi_n), np.sin(phi_n)])
    
    
    Z_dphi = 1j*omega*mu0 / (4*np.pi) * np.cos(phi_m - phi_n) * delta_r**2 * (ReGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - ReGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega) + 1j*ImGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - 1j*ReGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega))
    
    if np.sin(phi_n + phi_m) <= 1e-9 :
        Z_xy = 0
    else :
        Z_xy = 1j/(4*np.pi * omega * eps0) * np.sin(phi_m + phi_n) * delta_r**2 * (RederderXYGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - RederderXYGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega) + 1j*ImderderXYGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - 1j*ImderderXYGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega))
    
    if np.sin(phi_n) <= 1e-9 or np.sin(phi_m) <= 1e-9 :
        Z_y = 0
    else :       
        Z_y = 1j/(4*np.pi * omega * eps0) * np.sin(phi_n) * np.sin(phi_m) * delta_r**2 * (RederderYYGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - RederderYYGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega) + 1j*ImderderYYGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - 1j*ImderderYYGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega))

    if np.cos(phi_n) <= 1e-9 or np.cos(phi_m) <= 1e-9 :
        Z_x = 0
    else :
        Z_x = 1j/(4*np.pi * omega * eps0) * np.cos(phi_n) * np.cos(phi_m) * delta_r**2  * (RederderXXGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - RederderXXGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega) + 1j*ImderderXXGreen_function_noquad(1, r_m, r_n, dr_m, dr_n, omega) - 1j*ImderderXXGreen_function_noquad(0, r_m, r_n, dr_m, dr_n, omega))
    
    return Z_dphi+Z_x+Z_y+Z_xy

def calculate_impedance_noquad (R_block, angles, radii, delta_r, frequency):
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in range(0, len(R_block)):
        impedance_row = []
        for n in range (0, len(R_block)):
            if m == n or np.abs(angles[n]-angles[m]) <= 1e-9 :
                impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
                for i in range (len(R_block[m]) + len(R_block[n])):
                    impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_noquad(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
                    for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                        impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            else :
                for i in range(len(R_block[m])):
                    for j in range(len(R_block[n])):
                        impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn_noquad(m=m,n=n,i=max(0, len(R_block[m])-i-1),j=max(0, i-len(R_block[m])), angles=angles, radii=radii, R_block=R_block, delta_r=delta_r, omega=2*np.pi*frequency)
            impedance_row.append(impedance_mn)   
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance


def Zmn (m, n, i, j, angles, radii, R_block, delta_r, omega):
    
    phi_m, phi_n = angles[m], angles[n]
    a_m, a_n = radii[m], radii[n]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]

    if np.linalg.norm(r_m - r_n) < 3e-2 :
        return Zmn_double(m, n, i, j, angles, radii, R_block, delta_r, omega)
    else :
        return Zmn_single(m, n, i, j, angles, radii, R_block, delta_r, omega)
    

def calculate_impedance (R_block, angles, radii, delta_r, frequency):
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in tqdm(range(0, len(R_block))):
        impedance_row = []
        for n in range (0, len(R_block)):
            if m <= n :
                impedance_mn = np.zeros((len(R_block[m]), len(R_block[n])), dtype = complex)
                if m == n or np.abs(angles[n]-angles[m]) <= 1e-9 :
                    for i in range (len(R_block[m]) + len(R_block[n])):
                        impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn(m,n,max(0, len(R_block[m])-i-1),max(0, i-len(R_block[m])), angles, radii, R_block, delta_r, 2*np.pi*frequency)
                        for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                            impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
                else :
                    for i in range(len(R_block[m])):
                        for j in range(len(R_block[n])):
                            impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn(m,n,max(0, len(R_block[m])-i-1),max(0, i-len(R_block[m])), angles, radii, R_block, delta_r, 2*np.pi*frequency)
            else :
                impedance_mn = impedance_block[n][m].T
            impedance_row.append(impedance_mn)   
                
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance