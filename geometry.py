
import numpy as np

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
                radius = antenna.radius[m]
                segments_block_m.append(segment(position=position, tau=tau, radius=radius))
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
                        pos_k = start + delta_r * k * tau + delta_r * tau * pos_shift
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