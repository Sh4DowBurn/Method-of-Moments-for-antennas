import matrix_elements_2D as calc
import geometry_2D as gm
import numpy as np

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_currents (antenna, incident_field, frequency, delta_r):
    R_block, R = gm.calculate_positions(antenna=antenna, delta_r=delta_r)
    impedance = calc.calculate_impedance(antenna=antenna, R_block=R_block, delta_r=delta_r, frequency=frequency)
    voltage = calc.calculate_voltage(antenna=antenna, R_block=R_block, driven_voltage=incident_field, delta_r=delta_r)
    current = np.linalg.solve(impedance, voltage)
    return current

def directional_pattern (current, R, delta_r, frequency) :
    Radius = 1e2
    num_points = 1000
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = np.column_stack((Radius * np.cos(angles), Radius * np.sin(angles)))
    
    k = 2 * np.pi * frequency / light_speed
    E = []
    for point in points:
        E_i = 0
        for j in range(len(current)):
            rmn = np.linalg.norm(point - R[j])
            E_i += current[j] * np.exp(-1j * k * rmn) / rmn
        E_i = -1j * frequency * mu0 * delta_r * np.abs(E_i) / 2
        E.append(E_i)
    E_total = np.abs(E)
    return E_total, angles