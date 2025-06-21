#! This is a solver for Electric Field Integral Equation (EFIE)

import numpy as np 
from scipy import linalg 
import scipy.integrate as integrate 
from tqdm import tqdm
import geometry as gm
import matrix_elements as matrix_elements

# define constants

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

# function for calculating current amplitudes
def calc_current_amplitudes (structure_type, basis_functions, antenna, frequency, delta_r):

    # discretisation of parametric antenna by segments
    if structure_type == 'yagi-uda':
        segments_block, source_segments = gm.yagi_to_segments(antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)
    elif structure_type == 'tree':
        segments_block, source_segments = gm.tree_to_segments(antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)
        
    # solving matrix equation to find current vector
    voltage = matrix_elements.calculate_voltage(basis_functions=basis_functions, segments_block=segments_block, source_segments=source_segments, delta_r=delta_r)
    impedance = matrix_elements.calculate_impedance(basis_functions=basis_functions, structure_type=structure_type, segments_block=segments_block, frequency=frequency, delta_r=delta_r)
    current = np.linalg.solve(impedance, voltage)

    # define an array of position and amplitudes of current
    R, curr = [], []
    curr_pos = 0
    for m in range(len(segments_block)):
        for i in range(len(segments_block[m])):
            R.append(segments_block[m][i].position)
            curr.append(current[curr_pos])
            curr_pos += 1
    current, R = np.array(curr), np.array(R)
    
    return current, R
    
def calc_field_pattern (phi, theta, basis_functions, structure_type, antenna, current, R, delta_r, frequency) :

    omega = 2 * np.pi * frequency
    
    v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    aux_v = np.array([1, 0, 0])
    perp_v1 = np.cross(v, aux_v) / np.linalg.norm(np.cross(v, aux_v))
    perp_v2 = np.cross(perp_v1, v) / np.linalg.norm(np.cross(perp_v1, v))

    Radius, num_points = 1e2, 100
    angles = np.linspace(0, 2 * np.pi, num_points)

    points = np.zeros((num_points, 3))
    for i, angle in enumerate(angles):
        points[i] = Radius * (np.cos(angle) * perp_v1 + np.sin(angle) * perp_v2)

    if structure_type == 'yagi-uda':
        segments_block, source_segments = gm.yagi_to_segments(antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)
    elif structure_type == 'tree':
        segments_block, source_segments = gm.tree_to_segments(antenna=antenna, basis_functions=basis_functions, delta_r=delta_r)

    if basis_functions == 'pulse_optimized':
        E = []
        k = 2 * np.pi * frequency / light_speed
        for point in points:
            E_i = 0
            for j in range(len(R)):
                rmn = np.linalg.norm(point - R[j])
                E_i += current[j] * np.exp(-1j * k * rmn) / rmn
            E_i = -1j * frequency * mu0 * delta_r * np.abs(E_i) / 2
            E.append(E_i)
        E_total = np.abs(E)
        return E_total, angles
        
    E = []
    for point in points:
        curr_pos = 0
        if basis_functions == 'triangle':
            t_min, t_max = -1/2, 3/2
        elif basis_functions == 'pulse':
            t_min, t_max = 0, 1
        E_i = 0
        for m in range(len(segments_block)):
            for i in range(len(segments_block[m])):
                a_m, tau_m, r_m = segments_block[m][i].radius, segments_block[m][i].tau, segments_block[m][i].position
                dr_m = delta_r * tau_m
                E_i += -1j * omega * mu0 * delta_r / (4 * np.pi) * current[curr_pos] * (integrate.quad(exp_dp_real, t_min, t_max, args=(point, r_m, dr_m, omega, basis_functions))[0] + 1j * integrate.quad(exp_dp_imag, t_min, t_max, args=(point, r_m, dr_m, omega, basis_functions))[0])
                curr_pos += 1
        E.append(E_i)
    E_total = np.abs(E)
    
    return E_total, angles
        
        