#! This is a solver for Electric Field Integral Equation (EFIE)

import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm
import geometry as gm
import matrix_elements as matrix_elements

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12
c = light_speed

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

def exp_dp_real (t_m, point, r_m, dr_m, omega, basis_functions):
    k = omega/c
    rmn = np.linalg.norm(point - r_m - dr_m * (t_m - 1/2))
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_m * np.cos(-k * rmn) / rmn

def exp_dp_imag (t_m, point, r_m, dr_m, omega, basis_functions):
    k = omega/c
    rmn = np.linalg.norm(point - r_m - dr_m * (t_m - 1/2))
    f_m = weight_func(basis_functions=basis_functions, t_m=t_m, r_m=r_m, dr_m=dr_m)
    return f_m * np.sin(-k * rmn) / rmn

def calc_current_amplitudes (structure_type, basis_functions, antenna, frequency, delta_r):

    segments_block, source_segments = gm.antenna_to_segments(structure_type=structure_type, antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)
    
    voltage = matrix_elements.calculate_voltage(basis_functions=basis_functions, segments_block=segments_block, source_segments=source_segments, delta_r=delta_r)
    impedance = matrix_elements.calculate_impedance(basis_functions=basis_functions, structure_type=structure_type, segments_block=segments_block, frequency=frequency, delta_r=delta_r)
    current = np.linalg.solve(impedance, voltage)

    R, curr = [], []
    curr_pos = 0
    for m in range(len(segments_block)):
        for i in range(len(segments_block[m])):
            R.append(segments_block[m][i].position)
            curr.append(current[curr_pos])
            curr_pos += 1
    current, R = np.array(curr), np.array(R)
    
    return current, R, impedance, voltage, segments_block, source_segments
    
def calc_field_pattern (phi, theta, distance, basis_functions, structure_type, antenna, current, R, delta_r, frequency) :

    omega = 2 * np.pi * frequency
    k = omega / light_speed
    v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    aux_v = np.array([1, 0, 0])
    perp_v1 = np.cross(v, aux_v) / np.linalg.norm(np.cross(v, aux_v))
    perp_v2 = np.cross(perp_v1, v) / np.linalg.norm(np.cross(perp_v1, v))

    Radius, num_points = distance, 100
    angles = np.linspace(0, 2 * np.pi, num_points)

    points = np.zeros((num_points, 3))
    for i, angle in enumerate(angles):
        points[i] = Radius * (np.cos(angle) * perp_v1 + np.sin(angle) * perp_v2)

    segments_block, source_segments = gm.antenna_to_segments(structure_type=structure_type, antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)
    
    E = []
    for point in points:
        curr_pos = 0
        E_i = 0
        if basis_functions == 'pulse':
            for m in range(len(segments_block)):
                for i in range(len(segments_block[m])):
                    a_n, tau_n, r_n = segments_block[m][i].radius, segments_block[m][i].tau, segments_block[m][i].position
                    dr_n = delta_r * tau_n
                    rmn = np.linalg.norm(point - r_n)
                    k_vec = (point - r_n) / rmn
                    k_p = k * np.dot(k_vec, dr_n)
                    if k_p == 0:
                        E_i += - 1j * omega * mu0 * current[curr_pos] * delta_r / (4 * np.pi * rmn) * np.exp(-1j * k *rmn)
                    else:
                        E_i += - omega * mu0 * current[curr_pos] * delta_r / (4*np.pi*rmn*k_p) * np.exp(-1j*k*rmn) * np.exp(-1j*k_p/2) * (np.exp(1j*k_p) - 1)
                    curr_pos += 1
        elif basis_functions == 'triangle':
            for m in range(len(segments_block)):
                for i in range(len(segments_block[m])):
                    a_n, tau_n, r_n = segments_block[m][i].radius, segments_block[m][i].tau, segments_block[m][i].position
                    dr_n = delta_r * tau_n
                    rmn = np.linalg.norm(point - r_n)
                    k_vec = (point - r_n) / rmn
                    k_p = k * np.dot(k_vec, dr_n)
                    if k_p == 0:
                        E_i += - 1j * omega * mu0 * current[curr_pos] * delta_r / (2 * np.pi * rmn) * np.exp(-1j * k *rmn)
                    else:
                        E_i += - 1j * omega * mu0 * current[curr_pos] * delta_r / (4*np.pi*rmn*k_p**2) * np.exp(-1j*k*rmn) * np.exp(-1j*k_p) * (2*np.exp(1j*k_p) - np.exp(2j*k_p) - 1)
                    curr_pos += 1
        E.append(E_i)
    E_total = np.abs(E)
    
    return E_total, angles
        
        