import matrix_elements_2D as calc
import geometry_2D as gm
import numpy as np

def directional_pattern(antenna, incident_voltage, frequency, delta_r):
    R_block, R = gm.calculate_positions(antenna=antenna, delta_r=delta_r)
    impedance = calc.calculate_impedance(antenna=antenna, R_block=R_block, delta_r=delta_r, frequency=frequency)
    voltage = calc.calculate_voltage(antenna=antenna, R_block=R_block, driven_voltage=incident_voltage, delta_r=delta_r)
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
    return current_block, current