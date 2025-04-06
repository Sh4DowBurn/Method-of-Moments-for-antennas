import numpy as np
import pyswarms as ps
import geometry_2D as gm
import EFIE_solver_2D as efie
import matplotlib.pyplot as plt 

class antenna:
    def __init__(self, position, angle, length, source_position, radius):
        self.position = position
        self.angle = angle
        self.length = length
        self.source_position = source_position
        self.radius = radius

light_speed, mu0, eps0, incident_field, frequency = 299792458., 4*np.pi*1e-7, 8.854e-12, 10, 1e6 * 146
omega = 2 * np.pi * frequency
radius = 3.175e-3
source_positions = np.array([[0.0, 0.0]])
delta_r = light_speed / frequency / 1e2

k_form, k_max, k_side = 1, 1, 1
E_0 = 0.5

phi = np.linspace(0, 2*np.pi, 1000)
aim_func = np.zeros(len(phi))
for i in range(len(phi)): aim_func[i] = max(0,np.sin(np.pi/2-phi[i]))**7 

def graph_to_antenna(graph, structure_type):
    center_positions, angles, lengths, radii = np.zeros((len(graph)-1,2)), np.zeros(len(graph)-1), np.zeros(len(graph)-1), np.zeros(len(graph)-1)
    source = source_positions
    if structure_type == 'polygonal chain':
        for i in range (len(graph)-1):
            center_positions[i] = (graph[i] + graph[i+1]) / 2
            dy = graph[i+1, 1] - graph[i, 1]
            dx = graph[i+1, 0] - graph[i, 0]
            angles[i] = np.arctan2(dy, dx)
            lengths[i] = np.linalg.norm(graph[i+1] - graph[i])
            radii[i] = radius
    ans = antenna(center_positions, angles, lengths, source, radii)
    return ans
        
def check_self_intersection(graph):
    n = len(graph)
    for i in range(n):
        for j in range(i + 2, n):
            p1, q1 = graph[i], graph[(i + 1) % n]
            p2, q2 = graph[j], graph[(j + 1) % n]
            
            o1 = (q1[1] - p1[1]) * (p2[0] - q1[0]) - (q1[0] - p1[0]) * (p2[1] - q1[1])
            o2 = (q1[1] - p1[1]) * (q2[0] - q1[0]) - (q1[0] - p1[0]) * (q2[1] - q1[1])
            o3 = (q2[1] - p2[1]) * (p1[0] - q2[0]) - (q2[0] - p2[0]) * (p1[1] - q2[1])
            o4 = (q2[1] - p2[1]) * (q1[0] - q2[0]) - (q2[0] - p2[0]) * (q1[1] - q2[1])
            if (o1 != o2 and o3 != o4):
                return True
            if o1 == 0 and (p2[0] <= max(p1[0], q1[0]) and p2[0] >= min(p1[0], q1[0]) and p2[1] <= max(p1[1], q1[1]) and p2[1] >= min(p1[1], q1[1])):
                return True
            if o2 == 0 and (q2[0] <= max(p1[0], q1[0]) and q2[0] >= min(p1[0], q1[0]) and q2[1] <= max(p1[1], q1[1]) and q2[1] >= min(p1[1], q1[1])):
                return True
            if o3 == 0 and (p1[0] <= max(p2[0], q2[0]) and p1[0] >= min(p2[0], q2[0]) and p1[1] <= max(p2[1], q2[1]) and p1[1] >= min(p2[1], q2[1])):
                return True
            if o4 == 0 and (q1[0] <= max(p2[0], q2[0]) and q1[0] >= min(p2[0], q2[0]) and q1[1] <= max(p2[1], q2[1]) and q1[1] >= min(p2[1], q2[1])):
                return True
    return False


def fit_form (E_total):
    return np.clip(1 - np.dot(E_total/np.max(E_total),aim_func)/np.linalg.norm(aim_func)/np.linalg.norm(E_total/np.max(E_total)),0, 1e9)

def fit_max (E_total):
    return np.clip(np.exp(1 - E_total[0]/E_0) - 1, 0, 1e9)

def fit_side (E_total):
    return np.clip(np.exp(np.max(E_total[int(len(phi)/4):int(len(phi)*3/4)])/E_total[0]) - 1, 0, 1e9)

def my_objective_function(solution):
    fit = np.zeros(len(solution))
    for s in range(len(fit)):
        graph = np.concatenate(([[0.0,0.0]], solution[s].reshape(vertex_num, 2)))
        if not check_self_intersection(graph=graph) :
            fit[s] = 1e9
        else :
            antenna = graph_to_antenna(graph, structure_type)
            _, R = gm.calculate_positions(antenna, delta_r)
            current = efie.calculate_currents(antenna, incident_field, frequency, delta_r) 
            E_total, angles = efie.directional_pattern(current, R, delta_r, frequency)
            plt.polar(angles,E_total)
            fit[s] = k_form * fit_form(E_total) + k_side * fit_side(E_total) + k_max * fit_max(E_total)
            data_currents.append(current)
            data_positions.append(R)
    return fit

data_currents = []
data_positions = []

options = {
    'c1': 0.9,  
    'c2': 0.5,  
    'w': 0.8  
}

structure_type = 'polygonal chain'
vertex_num = 5
dimensions = vertex_num * 2
n_particles = 2
wavelength = light_speed / frequency
iters = 1

low_bound = -wavelength/2
up_bound = wavelength/2
lower_bounds = np.full(dimensions, low_bound)
upper_bounds = np.full(dimensions, up_bound)
bounds = (lower_bounds, upper_bounds)

optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)

best_cost, best_pos = optimizer.optimize(my_objective_function, iters=iters)

print("Лучшее сходство:", best_cost)
print("Лучшее решение:", best_pos)

np.savez(
    f'data_{structure_type.replace(" ", "_")}_{vertex_num}vertices.npz',
    opt=np.array([best_pos, best_cost], dtype=object),
    params=np.array([options["c1"], options["c2"], options["w"], n_particles, iters], dtype=object),
    current=np.array(data_currents, dtype=object),
    position=np.array(data_positions, dtype=object),
    allow_pickle=True
)