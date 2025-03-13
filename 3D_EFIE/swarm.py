import numpy as np
import pyswarms as ps
import geometry as gm
import matrix_elements as matrix
import EFIE_solver as efie
light_speed, mu0, eps0, incident_voltage, frequency = 299792458., 4*np.pi*1e-7, 8.854e-12, 10, 1e6 * 146
omega = 2 * np.pi * frequency

delta_r = light_speed / frequency / 1e2

k_form, k_max, k_side = 1, 1, 1
E_0 = 0.4 

phi = np.linspace(1e-6, 2*np.pi-1e-6, 1000)
aim_func = np.zeros(len(phi))
for i in range(len(phi)): aim_func[i] = max(0,np.sin(np.pi/2-phi[i]))**7 

def fit_form (E_total):
    return np.clip(1 - np.dot(E_total/np.max(E_total),aim_func)/np.linalg.norm(aim_func)/np.linalg.norm(E_total/np.max(E_total)),0, 1e9)

def fit_max (E_total):
    return np.clip(np.exp(1 - E_total[0]/E_0) - 1, 0, 1e9)

def fit_side (E_total):
    return np.clip(np.exp(np.max(E_total[int(len(phi)/4):int(len(phi)*3/4)])/E_total[0]) - 1, 0, 1e9)

def my_objective_function(solution):
    fit = np.zeros(len(solution))
    radius = 3.175e-3
    for s in range(len(fit)):
        antenna = np.concatenate(([[0.0,0.0,0.0]], solution[s].reshape(num_elements, 3)))
        source_position = np.array([(antenna[1]+antenna[2])/2])
        R_block, R = gm.calculate_positions(antenna, delta_r)
        impedance = matrix.calculate_impedance(antenna, R_block, delta_r, radius, frequency)
        voltage_block, voltage = matrix.calculate_voltage(R_block, incident_voltage, source_position, delta_r)
        current_block, current, E_total, phi = efie.calculate_directional_pattern(antenna, source_position, incident_voltage, frequency, radius, delta_r)
        fit[s] = k_form * fit_form(E_total) + k_side * fit_side(E_total) + k_max * fit_max(E_total)
        data_currents.append(current)
        data_positions.append(R)
    return fit

options = {
    'c1': 0.9,  
    'c2': 0.5,  
    'w': 0.8  
}

data_currents = []
data_positions = []

num_elements = 5
dimensions = num_elements * 3
wavelength = light_speed / frequency

optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions, options=options)

best_cost, best_pos = optimizer.optimize(my_objective_function, iters=1)

print("Лучшее сходство:", best_cost)
print("Лучшее решение:", best_pos)

np.savez('data1.npz', current = np.array(data_currents, dtype=object), position = np.array(data_positions, dtype = object))