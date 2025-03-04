
import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 

c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def ReGreen_function(t_m, t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega):
    dr = np.linalg.norm(dr_n)
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return (np.cos(phi_m-phi_n) * np.linalg.norm(dr_m) * dr * np.exp(-1j * omega/c * rmn)/rmn).real

def ImGreen_function(t_m, t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega):
    dr = np.linalg.norm(dr_n)
    rmn = np.linalg.norm(r_m - r_n + dr_m*(t_m-1/2) - dr_n*(t_n-1/2))
    return (np.cos(phi_m-phi_n) * np.linalg.norm(dr_m) * dr * np.exp(-1j * omega/c * rmn)/rmn).imag

def RederXGreen_function(t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega):
    rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    polypartp = -(r_m[0] + dr_m[0]/2 - r_n[0] - dr_n[0] * (t_n - 1/2))
    polypartm = (r_m[0] - dr_m[0]/2 - r_n[0] - dr_n[0] * (t_n - 1/2))
    dr = np.linalg.norm(dr_n)
    return (dr*(polypartp * (1 + 1j * omega/c * rmnp) / rmnp**3 * np.exp(-1j * omega/c * rmnp) + polypartm * (1 + 1j * omega/c * rmnm) / rmnm**3 * np.exp(-1j * omega/c * rmnm))).real

def ImderXGreen_function(t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega):
    rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    polypartp = -(r_m[0] + dr_m[0]/2 - r_n[0] - dr_n[0] * (t_n - 1/2))
    polypartm = (r_m[0] - dr_m[0]/2 - r_n[0] - dr_n[0] * (t_n - 1/2))
    dr = np.linalg.norm(dr_n)
    return (dr*(polypartp * (1 + 1j * omega/c * rmnp) / rmnp**3 * np.exp(-1j * omega/c * rmnp) + polypartm * (1 + 1j * omega/c * rmnm) / rmnm**3 * np.exp(-1j * omega/c * rmnm))).imag

def RederYGreen_function(t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, x0, y0, omega):
    
    if np.abs(np.tan(phi_m)) > 1e9 :
        ymp, ymm = r_m[1] + dr_m[1]/2, r_m[1] - dr_m[1]/2
        xmp, xmm = (ymp - y0) / np.tan(phi_m) + x0, (ymm - y0) / np.tan(phi_m) + x0
        pp, pm = np.array([xmp, ymp]), np.array([xmm, ymm])
        polypartp = -((xmp - dr_n[0] - dr_n[0] * (t_n - 1/2))/np.tan(phi_m) + (ymp - dr_n[1] - dr_n[1] * (t_n - 1/2)))
        polypartm =  ((xmm - dr_n[0] - dr_n[0] * (t_n - 1/2))/np.tan(phi_m) + (ymm - dr_n[1] - dr_n[1] * (t_n - 1/2)))
    else :
        ymp, ymm = r_m[1] + dr_m[1]/2, r_m[1] - dr_m[1]/2
        xmp, xmm = x0, x0
        pp, pm = np.array([xmp, ymp]), np.array([xmm, ymm])
        polypartp = -((ymp - dr_n[1] - dr_n[1] * (t_n - 1/2)))
        polypartm =  ((ymm - dr_n[1] - dr_n[1] * (t_n - 1/2)))
    
    rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    
    #rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    #rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    #polypartp = -(r_m[1] + dr_m[1]/2 - r_n[1] - dr_n[1] * (t_n - 1/2))
    #polypartm = (r_m[1] - dr_m[1]/2 - r_n[1] - dr_n[1] * (t_n - 1/2))
    dr = np.linalg.norm(dr_n)
    return ((polypartp * (1 + 1j * omega/c * rmnp) / rmnp**3 * np.exp(-1j * omega/c * rmnp) + polypartm * (1 + 1j * omega/c * rmnm) / rmnm**3 * np.exp(-1j * omega/c * rmnm))).real

def ImderYGreen_function(t_n, r_m, r_n, dr_m, dr_n, phi_m, phi_n, x0, y0, omega):
    
    if np.abs(np.tan(phi_m)) > 1e9 :
        ymp, ymm = r_m[1] + dr_m[1]/2, r_m[1] - dr_m[1]/2
        xmp, xmm = (ymp - y0) / np.tan(phi_m) + x0, (ymm - y0) / np.tan(phi_m) + x0
        pp, pm = np.array([xmp, ymp]), np.array([xmm, ymm])
        polypartp = -((xmp - dr_n[0] - dr_n[0] * (t_n - 1/2))/np.tan(phi_m) + (ymp - dr_n[1] - dr_n[1] * (t_n - 1/2)))
        polypartm =  ((xmm - dr_n[0] - dr_n[0] * (t_n - 1/2))/np.tan(phi_m) + (ymm - dr_n[1] - dr_n[1] * (t_n - 1/2)))
    else :
        ymp, ymm = r_m[1] + dr_m[1]/2, r_m[1] - dr_m[1]/2
        xmp, xmm = x0, x0
        pp, pm = np.array([xmp, ymp]), np.array([xmm, ymm])
        polypartp = -((ymp - dr_n[1] - dr_n[1] * (t_n - 1/2)))
        polypartm =  ((ymm - dr_n[1] - dr_n[1] * (t_n - 1/2)))
    
    rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    
    #rmnp = np.linalg.norm(r_m + dr_m/2 - r_n - dr_n * (t_n - 1/2))
    #rmnm = np.linalg.norm(r_m - dr_m/2 - r_n - dr_n * (t_n - 1/2))
    #polypartp = -(r_m[1] + dr_m[1]/2 - r_n[1] - dr_n[1] * (t_n - 1/2))
    #polypartm = (r_m[1] - dr_m[1]/2 - r_n[1] - dr_n[1] * (t_n - 1/2))
    dr = np.linalg.norm(dr_n)
    return ((polypartp * (1 + 1j * omega/c * rmnp) / rmnp**3 * np.exp(-1j * omega/c * rmnp) + polypartm * (1 + 1j * omega/c * rmnm) / rmnm**3 * np.exp(-1j * omega/c * rmnm))).imag

def Zmn (m, n, i, j, antenna, R_block, delta_r, omega):

    phi_m, phi_n = antenna.angle[m], antenna.angle[n]
    a_m, a_n = antenna.radius[m], antenna.radius[n]
    x0, y0 = antenna.position[m]
    
    r_m = R_block[m][i] + np.array([-a_m*np.sin(phi_m), a_m*np.cos(phi_m)])
    r_n = R_block[n][j]
    
    dr_m = delta_r * np.array([np.cos(phi_m), np.sin(phi_m)])
    dr_n = delta_r * np.array([np.cos(phi_n), np.sin(phi_n)])
    
    
    Z_dphi = 1j*omega*mu0 / (4*np.pi) * (integrate.dblquad(ReGreen_function, 0, 1, lambda z1: 0, lambda z2: 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0] + 1j * integrate.dblquad(ImGreen_function, 0, 1, lambda x: 0, lambda x: 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0])
    
    if np.abs(np.cos(phi_m)) <= 1e-9 :
        Z_x = 0
        if np.abs(np.sin(phi_n+phi_m)) <= 1e-9 :
            Z_xy = 0
        else :
            Z_xy = 1j*mu0*c**2 / (4*np.pi*omega) * np.sin(phi_n + phi_m) / np.sin(phi_m) *  (integrate.quad(RederYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0] + 1j * integrate.quad(ImderYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0])
    
    else :
        Z_x = 1j*mu0*c**2 / (4*np.pi*omega) * np.cos(phi_n) * (integrate.quad(RederXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0] + 1j * integrate.quad(ImderXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0])
    
    if np.abs(np.sin(phi_m)) <= 1e-9 :
        Z_y = 0
        if np.abs(np.sin(phi_n+phi_m)) <= 1e-9 :
            Z_xy = 0
        else :
            Z_xy = 1j*mu0*c**2 / (4*np.pi*omega)* np.sin(phi_n + phi_m) / np.cos(phi_m) *  (integrate.quad(RederXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0] + 1j * integrate.quad(ImderXGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, omega))[0])
    else :
        Z_y = delta_r * 1j/(4*np.pi * omega * eps0) * np.sin(phi_n) * (integrate.quad(RederYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, x0, y0, omega))[0] + 1j * integrate.quad(ImderYGreen_function, 0, 1, args=(r_m, r_n, dr_m, dr_n, phi_m, phi_n, x0, y0, omega))[0])
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
            for i in range (len(R_block[m]) + len(R_block[n])):
                impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))] = Zmn(m,n,max(0, len(R_block[m])-i-1),max(0, i-len(R_block[m])),antenna, R_block, delta_r, 2*np.pi*frequency)
                for k in range (min( min(len(R_block[m]), len(R_block[n])), i+1, len(R_block[m]) + len(R_block[n]) - i)):
                    impedance_mn[max(0, len(R_block[m])-i-1) + k, max(0, i-len(R_block[m])) + k] = impedance_mn[max(0, len(R_block[m])-i-1), max(0, i-len(R_block[m]))]
            impedance_row.append(impedance_mn)
        impedance_block.append(impedance_row)
    print(element_num)
    impedance = np.zeros((sum(element_num),sum(element_num)), dtype = complex)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        for j in range (len(cum_n)-1):
            impedance[cum_n[i]:cum_n[i+1], cum_n[j]:cum_n[j+1]] = impedance_block[i][j]
    return impedance