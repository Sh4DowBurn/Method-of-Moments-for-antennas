import geometry as gm
import matrix_elements as matrix_elements
import numpy as np

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_directional_pattern(antenna, source_position, incident_voltage, frequency, radius, delta_r):
    
    R_block, R = gm.calculate_positions(antenna=antenna, delta_r=delta_r)
    voltage_block, voltage = matrix_elements.calculate_voltage(R_block=R_block, driven_voltage=incident_voltage, delta_r=delta_r, source_position=source_position)
    impedance = matrix_elements.calculate_impedance(antenna, R_block, delta_r, radius, frequency)
    current = np.linalg.solve(impedance, voltage)
    element_num = []
    for i in range (len(R_block)):
        element_num.append(len(R_block[i]))
    element_num = np.array(element_num)
    
    num_elements = sum(element_num)
    current_block = []
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        current_block.append(current[cum_n[i]:cum_n[i+1]])
        
    phi, theta = 0, 0
    v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    aux_v = np.array([1, 0, 0])
    perp_v1 = np.cross(v, aux_v)
    perp_v1 = perp_v1 / np.linalg.norm(perp_v1)
    perp_v2 = np.cross(perp_v1, v)
    perp_v2 = perp_v2 / np.linalg.norm(perp_v2) 

    Radius = 1e2
    num_points = 1000
    angles = np.linspace(0, 2 * np.pi, num_points)

    points = np.zeros((num_points, 3))
    for i, angle in enumerate(angles):
        points[i] = Radius * (np.cos(angle) * perp_v1 + np.sin(angle) * perp_v2)


    E = []
    k = 2*np.pi*frequency/light_speed
    for i in range (len(points)):
        E_i = 0
        for j in range (len(current)):
            rmn = np.linalg.norm(points[i] - R[j])
            E_i += current[j] * np.exp(-1j * k * rmn) / rmn
        E_i = -1j * frequency * mu0 * delta_r * np.abs(E_i) / 2
        E.append(E_i)
    E_total = np.abs(E)
    return current_block, current, E_total, angles