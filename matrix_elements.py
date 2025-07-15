import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm

c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def basis_func(basis_functions, t_n, r_n, dr_n):
    if basis_functions == 'pulse' :
        return 1
    elif basis_functions == 'triangle':
        if -1/2 <= t_n <= 1/2 :
            return 1/2 + t_n
        elif 1/2 <= t_n <= 3/2 :
            return 3/2 - t_n

def weight_func(basis_functions, t_m, r_m, dr_m):
    if basis_functions == 'pulse' :
        return 1
    elif basis_functions == 'triangle':
        if -1/2 <= t_m <= 1/2 :
            return 1/2 + t_m
        elif 1/2 <= t_m <= 3/2 :
            return 3/2 - t_m

def ReGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.cos(- k * rmn) / rmn
def ImGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.sin(- k * rmn) / rmn

def RederderXXGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real  
def ImderderXXGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
def ImderderYYGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_m * f_n * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderZZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).real
def ImderderZZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real
def ImderderXYGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def RederderYZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).real
def ImderderYZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).imag

def RederderXZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).real
def ImderderXZGreen_function_single(t_m, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    t_n = 1/2
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).imag


def Zmn_single (structure_type, basis_functions, m, n, i, j, segments_block, omega, delta_r):

    if basis_functions == 'pulse' :
        t_min, t_max = 0, 1
    elif basis_functions == 'triangle' :
        t_min, t_max = -1/2, 3/2
    
    a_m = segments_block[m][i].radius
    a_n = segments_block[n][j].radius
    
    tau_m = segments_block[m][i].tau
    tau_n = segments_block[n][j].tau
    
    r_m = segments_block[m][i].position
    r_n = segments_block[n][j].position

    if structure_type == 'tree':
        r_n = r_n + a_n * np.array([0,0,1])
    elif structure_type == 'yagi-uda':
        r_n = r_n + a_n * np.array([0,0,1])

    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    if np.abs(np.dot(tau_m, tau_n)) <= 1e-21:
        Z_0 = 0
    else:
        Z_0 = 1j*omega*mu0 / (4*np.pi) * np.dot(tau_m, tau_n) * delta_r**2 * (integrate.quad(ReGreen_function_single, t_min, t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImGreen_function_single, t_min, t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_m[0] * tau_m[0]) <= 1e-21:
        Z_xx = 0
    else:
        Z_xx = 1j/(4*np.pi * omega * eps0) * tau_m[0] * tau_n[0] * delta_r**2  * (integrate.quad(RederderXXGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderXXGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])

    if np.abs(tau_m[1] * tau_m[1]) <= 1e-21:
        Z_yy = 0
    else:
        Z_yy = 1j/(4*np.pi * omega * eps0) * tau_m[1] * tau_n[1] * delta_r**2  * (integrate.quad(RederderYYGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderYYGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_m[2] * tau_m[2]) <= 1e-21:
        Z_zz = 0
    else:
        Z_zz = 1j/(4*np.pi * omega * eps0) * tau_m[2] * tau_n[2] * delta_r**2  * (integrate.quad(RederderZZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderZZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0])  <= 1e-21:
        Z_xy = 0
    else:
        Z_xy = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) * delta_r**2  * (integrate.quad(RederderXYGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderXYGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) <= 1e-21:
        Z_xz = 0
    else:
        Z_xz = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) * delta_r**2  * (integrate.quad(RederderXZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderXZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) <= 1e-21:
        Z_yz = 0
    else:
        Z_yz = 1j/(4*np.pi * omega * eps0) * (tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) * delta_r**2  * (integrate.quad(RederderYZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.quad(ImderderYZGreen_function_single, t_min, t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    return Z_0 + Z_xx + Z_yy + Z_zz + Z_xy + Z_xz + Z_yz


def ReGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.cos(- k * rmn) / rmn
def ImGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.sin(- k * rmn) / rmn

def RederderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real  
def ImderderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
def ImderderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_m * f_n * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).real
def ImderderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real
def ImderderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def RederderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).real
def ImderderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).imag

def RederderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).real
def ImderderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).imag


def Zmn_double (structure_type, basis_functions, m, n, i, j, segments_block, omega, delta_r):

    if basis_functions == 'pulse' :
        t_min, t_max = 0, 1
    elif basis_functions == 'triangle' :
        t_min, t_max = -1/2, 3/2
        
    a_m = segments_block[m][i].radius
    a_n = segments_block[n][j].radius
    
    tau_m = segments_block[m][i].tau
    tau_n = segments_block[n][j].tau
    
    r_m = segments_block[m][i].position
    r_n = segments_block[n][j].position
    
    if structure_type == 'yagi-uda':
        r_n = r_n + a_n * np.array([0,0,1])
    elif structure_type == 'tree':
        r_n = r_n + a_n * np.array([0,0,1])   
        
    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    if np.abs(np.dot(tau_m, tau_n)) <= 1e-100:
        Z_0 = 0
    else:
        Z_0 = 1j*omega*mu0 / (4*np.pi) * np.dot(tau_m, tau_n) * delta_r**2 * (integrate.dblquad(ReGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_m[0] * tau_n[0]) <= 1e-100:
        Z_xx = 0
    else:
        Z_xx = 1j/(4*np.pi * omega * eps0) * tau_m[0] * tau_n[0] * delta_r**2  * (integrate.dblquad(RederderXXGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderXXGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])

    if np.abs(tau_m[1] * tau_n[1]) <= 1e-100:
        Z_yy = 0
    else:
        Z_yy = 1j/(4*np.pi * omega * eps0) * tau_m[1] * tau_n[1] * delta_r**2  * (integrate.dblquad(RederderYYGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderYYGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_m[2] * tau_n[2]) <= 1e-100:
        Z_zz = 0
    else:
        Z_zz = 1j/(4*np.pi * omega * eps0) * tau_m[2] * tau_n[2] * delta_r**2  * (integrate.dblquad(RederderZZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderZZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) <= 1e-100:
        Z_xy = 0
    else:
        Z_xy = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXYGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderXYGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) <= 1e-100:
        Z_xz = 0
    else:
        Z_xz = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderXZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    
    if np.abs(tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) <= 1e-100:
        Z_yz = 0
    else:
        Z_yz = 1j/(4*np.pi * omega * eps0) * (tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) * delta_r**2  * (integrate.dblquad(RederderYZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(ImderderYZGreen_function_double, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args = (r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])
    return (Z_0 + Z_xx + Z_yy + Z_zz + Z_xy + Z_xz + Z_yz)

def impedance_real(t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega / c
    
    tau_n = dr_n / np.linalg.norm(dr_n)
    tau_m = dr_m / np.linalg.norm(dr_m)
    
    c_0 = (np.dot(tau_m, tau_n))
    c_x = (1/k**2 * tau_n[0] * tau_m[0])
    c_y = (1/k**2 * tau_n[1] * tau_m[1])
    c_z = (1/k**2 * tau_n[2] * tau_m[2])
    c_xy = (1/k**2 * (tau_m[0]*tau_n[1] + tau_m[1]*tau_n[0]))
    c_xz = (1/k**2 * (tau_m[0]*tau_n[2] + tau_m[2]*tau_n[0]))
    c_yz = (1/k**2 * (tau_m[1]*tau_n[2] + tau_m[2]*tau_n[1]))

    C = c_x + c_y + c_z

    rmn =  (np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2)))
    dx = (r_m[0] - r_n[0] + dr_m[0]*(t_m-1/2) - dr_n[0]*(t_n-1/2))
    dy = (r_m[1] - r_n[1] + dr_m[1]*(t_m-1/2) - dr_n[1]*(t_n-1/2))
    dz = (r_m[2] - r_n[2] + dr_m[2]*(t_m-1/2) - dr_n[2]*(t_n-1/2))
    L = c_x * dx**2 + c_y * dy**2 + c_z * dz**2 + c_xy * dx * dy + c_xz * dx * dz + c_yz * dy * dz
    
    polypart1 = c_0 * rmn**2 - C + L * (3 - k**2 * rmn**2) / rmn**2
    polypart2 = k * (3 * L / rmn - rmn * C)
    
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    
    return f_n * f_m * (polypart1 * np.cos(k * rmn) + polypart2 * np.sin(k * rmn)) / rmn**3
def impedance_imag (t_m, t_n, r_m, r_n, dr_m, dr_n, omega, basis_functions):
    k = omega / c
    
    tau_n = dr_n / np.linalg.norm(dr_n)
    tau_m = dr_m / np.linalg.norm(dr_m)
    
    c_0 = (np.dot(tau_m, tau_n))
    c_x = (1/k**2 * tau_n[0] * tau_m[0])
    c_y = (1/k**2 * tau_n[1] * tau_m[1])
    c_z = (1/k**2 * tau_n[2] * tau_m[2])
    c_xy = (1/k**2 * (tau_m[0]*tau_n[1] + tau_m[1]*tau_n[0]))
    c_xz = (1/k**2 * (tau_m[0]*tau_n[2] + tau_m[2]*tau_n[0]))
    c_yz = (1/k**2 * (tau_m[1]*tau_n[2] + tau_m[2]*tau_n[1]))

    C = c_x + c_y + c_z

    rmn =  (np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2)))
    dx = (r_m[0] - r_n[0] + dr_m[0]*(t_m-1/2) - dr_n[0]*(t_n-1/2))
    dy = (r_m[1] - r_n[1] + dr_m[1]*(t_m-1/2) - dr_n[1]*(t_n-1/2))
    dz = (r_m[2] - r_n[2] + dr_m[2]*(t_m-1/2) - dr_n[2]*(t_n-1/2))
    L = c_x * dx**2 + c_y * dy**2 + c_z * dz**2 + c_xy * dx * dy + c_xz * dx * dz + c_yz * dy * dz
    
    polypart1 = c_0 * rmn**2 - C + L * (3 - k**2 * rmn**2) / rmn**2
    polypart2 = k * (3 * L / rmn - rmn * C)
    
    f_n = basis_func(basis_functions=basis_functions, t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    
    return f_n * f_m * (-polypart1 * np.sin(k * rmn) + polypart2 * np.cos(k * rmn)) / rmn**3

def Zmn (structure_type, basis_functions, m, n, i, j, segments_block, omega, delta_r):

    if basis_functions == 'pulse' :
        t_min, t_max = 0, 1
    elif basis_functions == 'triangle' :
        t_min, t_max = -1/2, 3/2
        
    a_m = segments_block[m][i].radius
    a_n = segments_block[n][j]. radius
    
    tau_m = segments_block[m][i].tau
    tau_n = segments_block[n][j].tau
    
    r_m = segments_block[m][i].position 
    r_n = segments_block[n][j].position 
    
    if structure_type == 'yagi-uda':
        r_n = r_n + a_n * np.array([0,0,1])
    elif structure_type == 'tree':
        r_n = r_n + a_n * np.array([0,0,1])

    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    return 1j*omega*mu0 / (4*np.pi) * delta_r **2 * (integrate.dblquad(impedance_real, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0] + 1j * integrate.dblquad(impedance_imag, t_min, t_max, lambda z1: t_min, lambda z2: t_max, args=(r_m, r_n, dr_m, dr_n, omega, basis_functions))[0])


def calculate_impedance (basis_functions, structure_type, segments_block, frequency, delta_r):
    
    omega = 2 * np.pi * frequency
    
    element_num = []
    for i in range (len(segments_block)):
        element_num.append(len(segments_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    
    if structure_type == 'yagi-uda':    
        for m in range(len(segments_block)):
            impedance_row = []
            for n in range(len(segments_block)):
                impedance_mn = np.zeros((len(segments_block[m]), len(segments_block[n])), dtype=complex)
                if m <= n :
                    for i in range (len(segments_block[m]) + len(segments_block[n])):
                        impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))] = Zmn(structure_type=structure_type,basis_functions=basis_functions, m=m, n=n, i=max(0, len(segments_block[m])-i-1), j=max(0, i-len(segments_block[m])), segments_block=segments_block, omega=2*np.pi*frequency, delta_r=delta_r)
                        for k in range (min( min(len(segments_block[m]), len(segments_block[n])), i+1, len(segments_block[m]) + len(segments_block[n]) - i)):
                            impedance_mn[max(0, len(segments_block[m])-i-1) + k, max(0, i-len(segments_block[m])) + k] = impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))]
                else :
                    impedance_mn = impedance_block[n][m].T
                impedance_row.append(impedance_mn)   
            impedance_block.append(impedance_row)
    
    elif structure_type == 'tree':
        for m in tqdm(range(len(segments_block))):
            impedance_row = []
            for n in range(len(segments_block)):
                impedance_mn = np.zeros((len(segments_block[m]), len(segments_block[n])), dtype=complex)
                tau_m = segments_block[m][0].tau
                tau_n = segments_block[n][0].tau
                if m <= n :
                    if m == n or np.linalg.norm(np.cross(tau_m, tau_n)) <= 1e-25:
                        for i in range (len(segments_block[m]) + len(segments_block[n])):
                            impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))] = Zmn(structure_type=structure_type,basis_functions=basis_functions, m=m, n=n, i=max(0, len(segments_block[m])-i-1), j=max(0, i-len(segments_block[m])), segments_block=segments_block, omega=2*np.pi*frequency, delta_r=delta_r)
                            for k in range (min( min(len(segments_block[m]), len(segments_block[n])), i+1, len(segments_block[m]) + len(segments_block[n]) - i)):
                                impedance_mn[max(0, len(segments_block[m])-i-1) + k, max(0, i-len(segments_block[m])) + k] = impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))]
                    else:
                        for i in range(len(segments_block[m])):
                            for j in range(len(segments_block[n])):
                                impedance_mn[i,j] = Zmn_single(structure_type,basis_functions,m,n,i,j,segments_block,omega,delta_r)
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

def calculate_voltage(basis_functions, segments_block, source_segments, delta_r):
    
    element_num = []
    for i in range (len(segments_block)):
        element_num.append(len(segments_block[i]))
    element_num = np.array(element_num)
    
    if basis_functions == 'pulse' :
        vol_shift, pos_shift = 1, 0
    elif basis_functions == 'triangle' :
        vol_shift, pos_shift = 0.5, -delta_r/2
    
    voltage_block = []
    for m in range(len(segments_block)):
        voltage_row = []
        for i in range(len(segments_block[m])):
            for k in range(len(source_segments)):
                if np.abs(np.linalg.norm(source_segments[k].position - segments_block[m][i].position) + pos_shift) <= 1e-8:
                    v_m = source_segments[k].field * vol_shift
                    voltage_row.append(v_m)
                else:
                    voltage_row.append(0.0)
        voltage_block.append(voltage_row)
    
    voltage = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        voltage[cum_n[i]:cum_n[i+1]] = voltage_block[i]
    
    return voltage