import geometry as gm
import matrix_elements as matrix
import numpy as np

def calculate_currents(antenna, source_position, incident_voltage, frequency, radius, delta_r):
    
    R_block, R = gm.calculate_positions(antenna=antenna, delta_r=delta_r)
    voltage_block, voltage = matrix.calculate_voltage(R_block=R_block, driven_voltage=incident_voltage, delta_r=delta_r, source_position=source_position)
    impedance = matrix.calculate_impedance(antenna, R_block, delta_r, radius, frequency)
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