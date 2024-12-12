# Import extension Better Comments for beauty
import better

#! This is a function for visualizating geometry of antenna

#* Import library for graphics
import matplotlib.pyplot as plt
import numpy as np

def plot_2dmodel (R, source_position, num_elements, delta_z):
    
    #* Define list of each segment
    segments = [0] * num_elements
    positions = np.array([R_m_i for R_m in R for R_m_i in R_m])
    for i in range (0, num_elements):
        segments[i] = ((positions[i, 1], positions[i, 2] - delta_z/2), (positions[i, 1], positions[i, 2] + delta_z/2))

    #* Create figure and axes
    fig, ax = plt.subplots()

    #* Plot the segments, where the source is located, in red, other in blue
    i = 0
    for start, end in segments:
        clr = 'blue'
        zorder = 10
        linewidth = 1
        if(all(positions[i] == source_position[0]) or all(positions[i-1] == source_position[0])):
            clr = 'red'
            zorder = np.inf
            linewidth = 3
        ax.plot([start[0], end[0]], [start[1], end[1]], color=clr, linewidth=linewidth, marker='.', zorder = zorder) 
        i += 1
    
    #* Set limits for axes 
    ax.set_xlim(min(positions[:,1])-0.1, max(positions[:,1])+0.1)
    ax.set_ylim(min(positions[:,2])-0.1, max(positions[:,2]+0.1))

    #* Name the axes and graph
    ax.set_title('2D Model of Yagi-Uda Antenna', size = 14)
    ax.set_xlabel('Y position, m', size = 12)
    ax.set_ylabel('Z position, m', size = 12)

    #* Add grid
    ax.grid(True)

    #* show figure
    plt.show()