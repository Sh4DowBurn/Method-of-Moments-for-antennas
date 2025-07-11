
import numpy as np

c, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

class segment:
    def __init__(self, position, tau, radius):
        self.position = position
        self.tau = tau
        self.radius = radius

class source:
    def __init__(self, position, field):
        self.position = position
        self.field = field

def antenna_to_segments(structure_type, antenna, basis_functions, delta_r):
    
    if structure_type == 'yagi-uda':
        
        element_num = np.round(antenna.length / delta_r).astype(int)
        for i in range(len(element_num)):
            if element_num[i] % 2 == 0:
                element_num += 1
        
        segments_block = []
        source_segments = []
    
        if basis_functions == 'pulse' :
            index_shift, pos_shift = 0, 0
        elif basis_functions == 'triangle' :
            index_shift, pos_shift = -1, 1/2
    
        for m in range(0, len(element_num)):
            segments_block_m = []
            theta, phi = antenna.angle[m]
            tau = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            for i in range(element_num[m] + index_shift):
                position = antenna.position[m] + tau * delta_r * (i - element_num[m]/2 + 1/2 + pos_shift)
                rad = antenna.radius[m]
                segments_block_m.append(segment(position=position, tau=tau, radius=rad))
            segments_block_m = np.array(segments_block_m)
            segments_block.append(segments_block_m)
    
        for m in range(len(antenna.source_position)):
            x, y, z, field = antenna.source_position[m]
            pos = np.array([x,y,z])
            source_segments.append(source(position=pos, field=field))
        
        source_segments = np.array(source_segments, dtype=object)
        segments_block = np.array(segments_block, dtype=object) 
    
    elif structure_type == 'tree':
        
        if basis_functions == 'pulse' :
            index_shift, pos_shift = 0, 0
        elif basis_functions == 'triangle' :
            index_shift, pos_shift = -1, 1/2
        
        its = antenna.its
        phi = antenna.phi
        radius = antenna.radius
        dr = antenna.length
        factor = antenna.f
        field = antenna.field
    
        segments_block = []
        pos = np.array([0.0, 0.0, 0.0])
        ang = np.array([0.0])
        pos_phi = 0
        
        R = []
        tau0 = np.array([1.0, 0.0, 0.0])
        n0 = int(dr / delta_r) + 1
        for k in range(n0 + index_shift):
            pos_k = delta_r * k * tau0 + delta_r * tau0 * pos_shift
            R.append(segment(position=pos_k, tau=tau0, radius=radius))
        R = np.array(R)
        segments_block.append(R)
        
        for _ in range(its):
            pos_new, ang_new = [], []
            for p, a in zip(pos, ang):
                ang1, ang2 = a + phi[pos_phi], a - phi[pos_phi+1]
                pos_phi += 2

                p1 = p + dr * np.array([np.cos(ang1), np.sin(ang1), 0.0])
                p2 = p + dr * np.array([np.cos(ang2), np.sin(ang2), 0.0])

                for start, end in [(p, p1), (p, p2)]:
                    length = np.linalg.norm(end - start)
                    n = int(length / delta_r) + 1
                    tau = (end - start) / length
                    R = []
                    for k in range(n + index_shift):
                        pos_k = start + delta_r * k * tau + delta_r * tau * pos_shift + np.array([antenna.length, 0.0, 0.0])
                        R.append(segment(position=pos_k, tau=tau, radius=radius))
                    R = np.array(R)
                    segments_block.append(R)

                pos_new.extend([p1, p2])
                ang_new.extend([ang1, ang2])

            dr *= factor
            pos = np.array(pos_new)
            ang = np.array(ang_new)

        segments_block = np.array(segments_block, dtype=object)
        source_segments = []
        sr = source(position=[0.0, 0.0, 0.0], field = field)
        source_segments.append(sr)
        source_segments = np.array(source_segments)
        
    return segments_block, source_segments

import matplotlib.pyplot as plt
def plot_antenna(structure_type, basis_functions, antenna, delta_r):
    segments_block, source_segments = antenna_to_segments(structure_type=structure_type, antenna=antenna, basis_functions=basis_functions,delta_r=delta_r)
    R = []
    for m in range(len(segments_block)):
        for i in range(len(segments_block[m])):
            R.append(segments_block[m][i].position)
    R = np.array(R)
    R_source = []
    for i in range(len(source_segments)):
        R_source.append(source_segments[i].position)
    R_source = np.array(R_source)
    plt.scatter(R[:,0], R[:,1], color = 'darkblue', s = 10, marker='o', zorder = 5)
    plt.scatter(R_source[:,0],R_source[:,1], color = 'red', s = 20, marker='o', label = 'source', zorder = 10)
    plt.legend()
    plt.title('Antenna geometry', size = 13)
    plt.xlabel('X position, m', size = 12)
    plt.ylabel('Y position, m', size = 12)
    plt.grid(zorder = 0)
    plt.axis('equal')
    
def plot_distribution(I, R, frequency):
    
    wavelength = c / frequency
    
    fig, ax = plt.subplots()
    sc = ax.scatter(R[:, 0] / wavelength, R[:, 1] / wavelength, c=np.abs(I), cmap='inferno', zorder = 5)
    ax.set_xlabel('X position, wavelength', size = 12)
    ax.set_ylabel('Y position, wavelength', size = 12)
    ax.set_title('Current distribution', size = 13)
    ax.grid(True, zorder = 0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Current amplitudes')

    plt.show()

    