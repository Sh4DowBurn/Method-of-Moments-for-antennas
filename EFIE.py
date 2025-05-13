import numpy as np
import geometry as gm
import matrix_elements as matrix_elements

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calc_current_distribution (structure_type, antenna, frequency, delta_r):
    
    if structure_type == 'yagi-uda':
        segments_block, source_segments = gm.yagi_to_segments(antenna=antenna, delta_r=delta_r)
        voltage = matrix_elements.calculate_voltage(segments_block=segments_block, source_segments=source_segments, delta_r=delta_r)
        impedance = matrix_elements.calculate_impedance(structure_type=structure_type, segments_block=segments_block, frequency=frequency, delta_r=delta_r)
        current = np.linalg.solve(impedance, voltage)
        
        R = []
        for m in range(len(segments_block)):
            for i in range(len(segments_block[m])):
                R.append(segments_block[m][i].position)
                
        return current, np.array(R)
    
def calc_directional_pattern (phi, theta, current, R, delta_r, frequency) :
    
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

    k = 2 * np.pi * frequency / light_speed
    E = []
    for point in points:
        E_i = 0
        for j in range(len(R)):
            rmn = np.linalg.norm(point - R[j])
            E_i += current[j] * np.exp(-1j * k * rmn) / rmn
        E_i = -1j * frequency * mu0 * delta_r * np.abs(E_i) / 2
        E.append(E_i)
    E_total = np.abs(E)
    return E_total, angles
        
        