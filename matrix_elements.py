import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm

c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def basis_func(t_n, r_n, dr_n):
    #Pulse
    #return 1
    #Triangle
    if t_n >= 0 and t_n <= 1/2 :
        return 2*t_n
    else :
        return 2 * (1 - t_n)

def weight_func(t_m, r_m, dr_m):
    #Pulse
    #return 1
    #Triangle
    if t_m >= 0 and t_m <= 1/2 :
        return 2*t_m
    else :
        return 2 * (1 - t_m)

def ReGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.cos(- k * rmn) / rmn
def ImGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_n * f_m * np.sin(- k * rmn) / rmn

def RederderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).real  
def ImderderXXGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    p2R_px2 = 1/R - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px ** 2 + polypart2 * p2R_px2) * np.exp(-1j * k * R)).imag

def RederderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).real
def ImderderYYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R 
    p2R_py2 = 1/R - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_m * f_n * (polypart1 * pR_py ** 2 + polypart2 * p2R_py2) * np.exp(-1j * k * R)).imag

def RederderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).real
def ImderderZZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R 
    p2R_pz2 = 1/R - (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2))**2 / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3)
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_pz ** 2 + polypart2 * p2R_pz2) * np.exp(-1j * k * R)).imag

def RederderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).real
def ImderderXYGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R 
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    p2R_pxy = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_py + polypart2 * p2R_pxy) * np.exp(-1j * k * R)).imag

def RederderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).real
def ImderderYZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_py = (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pyz = - (r_m[1] + dr_m[1] * (t_m - 1/2) - r_n[1] - dr_n[1] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_py * pR_pz + polypart2 * p2R_pyz) * np.exp(-1j * k * R)).imag

def RederderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).real
def ImderderXZGreen_function_double(t_m, t_n, r_m, r_n, dr_m, dr_n, omega):
    k = omega/c
    R = np.linalg.norm(r_m + dr_m * (t_m - 1/2) - r_n - dr_n * (t_n - 1/2))
    pR_px = (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) / R
    pR_pz = (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R
    p2R_pxz = - (r_m[0] + dr_m[0] * (t_m - 1/2) - r_n[0] - dr_n[0] * (t_n - 1/2)) * (r_m[2] + dr_m[2] * (t_m - 1/2) - r_n[2] - dr_n[2] * (t_n - 1/2)) / R**3
    polypart1 = -(1j * k / R**2 - (1 + 1j * k * R) * 1j * k / R**2 - 2 * (1 + 1j * k * R) / R**3) 
    polypart2 = -(1 + 1j * k * R) / R**2
    f_n = basis_func(t_n=t_n, r_n=r_n, dr_n=dr_n)
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return (f_n * f_m * (polypart1 * pR_px * pR_pz + polypart2 * p2R_pxz) * np.exp(-1j * k * R)).imag


def Zmn_double (m, n, i, j, segments_block, omega, delta_r):

    a_m = segments_block[m][i].radius
    
    tau_m = segments_block[m][i].tau
    tau_n = segments_block[n][j].tau
    
    r_m = segments_block[m][i].position + np.array([-a_m*tau_m[2], a_m*tau_m[1], a_m*tau_m[0]])
    r_n = segments_block[n][j].position

    dr_m = delta_r * tau_m
    dr_n = delta_r * tau_n
    
    Z_0 = 1j*omega*mu0 / (4*np.pi) * np.dot(tau_m, tau_n) * delta_r**2 * (integrate.dblquad(ReGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args=(r_m, r_n, dr_m, dr_n, omega))[0])
    
    Z_x = 1j/(4*np.pi * omega * eps0) * tau_m[0] * tau_n[0] * delta_r**2  * (integrate.dblquad(RederderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXXGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])

    Z_y = 1j/(4*np.pi * omega * eps0) * tau_m[1] * tau_n[1] * delta_r**2  * (integrate.dblquad(RederderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderYYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    Z_z = 1j/(4*np.pi * omega * eps0) * tau_m[2] * tau_n[2] * delta_r**2  * (integrate.dblquad(RederderZZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderZZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    Z_xy = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[1] + tau_n[1] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXYGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    Z_xz = 1j/(4*np.pi * omega * eps0) * (tau_n[0] * tau_m[2] + tau_n[2] * tau_m[0]) * delta_r**2  * (integrate.dblquad(RederderXZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderXZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    Z_yz = 1j/(4*np.pi * omega * eps0) * (tau_n[2] * tau_m[1] + tau_n[1] * tau_m[2]) * delta_r**2  * (integrate.dblquad(RederderYZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0] + 1j * integrate.dblquad(ImderderYZGreen_function_double, 0, 1, lambda z1: 0, lambda z2: 1, args = (r_m, r_n, dr_m, dr_n, omega))[0])
    
    return Z_0 + Z_x + Z_y + Z_z + Z_xy + Z_xz + Z_yz

def Zmn (m, n, i, j, segments_block, omega, delta_r):
    return Zmn_double(m=m, n=n, i=i, j=j, segments_block=segments_block, omega=omega, delta_r=delta_r)


def calculate_impedance (structure_type, segments_block, frequency, delta_r):
    
    element_num = []
    for i in range (len(segments_block)):
        element_num.append(len(segments_block[i]))
    element_num = np.array(element_num)
    
    impedance_block = []
    for m in  tqdm(range(0, len(segments_block))):
        impedance_row = []
        for n in range (0, len(segments_block)):
            impedance_mn = np.zeros((len(segments_block[m]), len(segments_block[n])), dtype=complex)
            if structure_type == 'yagi-uda' :
                if m <= n :
                    for i in range (len(segments_block[m]) + len(segments_block[n])):
                        impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))] = Zmn(m=m, n=n, i=max(0, len(segments_block[m])-i-1), j=max(0, i-len(segments_block[m])), segments_block=segments_block, omega=2*np.pi*frequency, delta_r=delta_r)
                        for k in range (min( min(len(segments_block[m]), len(segments_block[n])), i+1, len(segments_block[m]) + len(segments_block[n]) - i)):
                            impedance_mn[max(0, len(segments_block[m])-i-1) + k, max(0, i-len(segments_block[m])) + k] = impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))]
                else :
                    impedance_mn = impedance_block[n][m].T
            #if m == n:
            #    for i in range (len(segments_block[m]) + len(segments_block[n])):
            #        impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))] = Zmn(m=m, n=n, i=max(0, len(segments_block[m])-i-1), j=max(0, i-len(segments_block[m])), segments_block=segments_block, omega=2*np.pi*frequency, delta_r=delta_r)
            #        for k in range (min( min(len(segments_block[m]), len(segments_block[n])), i+1, len(segments_block[m]) + len(segments_block[n]) - i)):
            #            impedance_mn[max(0, len(segments_block[m])-i-1) + k, max(0, i-len(segments_block[m])) + k] = impedance_mn[max(0, len(segments_block[m])-i-1), max(0, i-len(segments_block[m]))]
            #elif m < n :
            #    impedance_mn = np.zeros((len(segments_block[m]), len(segments_block[n])), dtype = complex)
            #    for i in range(len(segments_block[m])):
            #        for j in range(len(segments_block[n])):
            #            impedance_mn[i,j] = Zmn(m=m, n=n, i=i, j=j, segments_block=segments_block, omega=2*np.pi*frequency, delta_r=delta_r)   
            #elif m > n :
            #    impedance_mn = impedance_block[n][m].T
            
            impedance_row.append(impedance_mn)   
        impedance_block.append(impedance_row)
    
    num_elements = sum(element_num)
    impedance = np.zeros((num_elements, num_elements), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]

    return impedance

def vol_func (t_m, r_m, dr_m, voltage):
    f_m = weight_func(t_m=t_m, r_m=r_m, dr_m=dr_m)
    return voltage * f_m
    

def calculate_voltage(segments_block, source_segments, delta_r):
    
    element_num = []
    for i in range (len(segments_block)):
        element_num.append(len(segments_block[i]))
    element_num = np.array(element_num)
    
    voltage_block = []
    for m in range(len(segments_block)):
        voltage_row = []
        for i in range(len(segments_block[m])):
            for k in range(len(source_segments)):
                if np.abs(np.linalg.norm(source_segments[k].position - segments_block[m][i].position) - delta_r/2) <= 1e-8:
                    v_m = integrate.quad(vol_func, 0, 1, args=(segments_block[m][i].position, delta_r * segments_block[m][i].tau, source_segments[k].field))[0]
                    voltage_row.append(v_m)
                else:
                    voltage_row.append(0.0)
        voltage_block.append(voltage_row)
    
    voltage = np.zeros((sum(element_num)), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        voltage[cum_n[i]:cum_n[i+1]] = voltage_block[i]
    
    return voltage