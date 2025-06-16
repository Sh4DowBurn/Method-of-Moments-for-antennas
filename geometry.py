
import numpy as np

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

class segment:
    def __init__(self, position, tau, radius):
        self.position = position
        self.tau = tau
        self.radius = radius

class source:
    def __init__(self, position, field):
        self.position = position
        self.field = field

def yagi_to_segments(antenna, basis_functions, delta_r):
    
    element_num = np.round(antenna.length / delta_r).astype(int)
    if np.round(antenna.length / delta_r).astype(int)%2 == 0:
        element_num += 1
        
    segments_block = []
    source_segments = []
    
    if basis_functions == 'pulse' :
        
        for m in range(0, len(element_num)):
            segments_block_m = []
            theta, phi = antenna.angle[m]
            tau = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            for i in range(element_num[m]):
                position = antenna.position[m] + tau * delta_r * (i - element_num[m]/2 + 1/2)
                radius = antenna.radius[m]
                segments_block_m.append(segment(position=position, tau=tau, radius=radius))
            segments_block_m = np.array(segments_block_m)
            segments_block.append(segments_block_m)
    
        for m in range(len(antenna.source_position)):
            x, y, z, field = antenna.source_position[m]
            pos = np.array([x,y,z])
            source_segments.append(source(position=pos, field=field))
    
    elif basis_functions == 'triangle' :
    
        for m in range(0, len(element_num)):
            segments_block_m = []
            theta, phi = antenna.angle[m]
            tau = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            for i in range(element_num[m]-1):
                position = antenna.position[m] + tau * delta_r * (i - element_num[m]/2 + 1)
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
    
    return segments_block, source_segments