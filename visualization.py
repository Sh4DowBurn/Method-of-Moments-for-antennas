# Import extension Better Comments for beauty
import better

#! This is a function for visualizating geometry of antenna

#* Import library for graphics
import matplotlib.pyplot as plt
import numpy as np
import colormaps as cmaps
from matplotlib import colormaps as plt_cmaps

#* define constants
light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12 

def plot_2dmodel (R, source_position, num_elements, delta_z):

    #* Create figure and axes
    fig, ax = plt.subplots()
    plt.errorbar(np.array(R[:,1]),np.array(R[:,2]), xerr = delta_z, linestyle = "none", marker = '.', markersize = '2', zorder = 10, color = 'blue')
    for i in range (len(R)):
        for k in range (len(source_position)):
            if all(source_position[k] == R[i]):
                plt.errorbar(np.array(R[i,1]),np.array(R[i,2]), xerr = delta_z, linestyle = "none", marker = '.', markersize = '5', zorder = np.inf, color = 'red')
    #* Set limits for axes 
    ax.set_xlim(min(R[:,1])-0.1, max(R[:,1])+0.1)
    ax.set_ylim(min(R[:,2])-0.1, max(R[:,2]+0.1))

    #* Name the axes and graph
    ax.set_title('2D Model of Yagi-Uda Antenna', size = 14)
    ax.set_xlabel('Y position, m', size = 12)
    ax.set_ylabel('Z position, m', size = 12)

    #* Add grid
    ax.grid(True)

    #* show figure
    plt.show()

def calculate_positions (element_length, element_position, frequency, delta_z) :
    
    #* Calculate some parametres of system
    wavelength, wavenumber = light_speed / frequency, 2 * np.pi * frequency / light_speed
    
    #* calculating number of sigments on each element
    element_num  = np.zeros(len(element_length), dtype = int)
    for i in range (len(element_length)):
        element_num[i] = int(element_length[i]/delta_z) if int(element_length[i]/delta_z)%2!=0 else int(element_length[i]/delta_z)+1
        
    #* Define list of positions, where R[m][i] - the position of i-th segment on m-th element
    R_block = [0] * len(element_num)
    for m in range (0, len(element_num)):
        R_block[m] = np.zeros((element_num[m], 3))
        for i in range (0, len(R_block[m])):
            R_block[m][i, 0] = 0
            R_block[m][i, 1] = element_position[m]
            R_block[m][i, 2] = -element_num[m]*delta_z / 2 + delta_z * (1/2 + i)
    
    #* Deploying a block matrix (reshape)
    num_elements = sum(element_num)
    R = np.zeros((num_elements,3), dtype = float)
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n)-1):
        R[cum_n[i]:cum_n[i+1]] = R_block[i]
    return element_num, R_block, R

def plot_together (R_block, element_currents) :
    cmap = plt_cmaps['hot']
    for i in range (len(R_block)):
        plt.plot(R_block[i][:,2], np.abs(element_currents[i])*1000, zorder = np.inf, label = f'element {1+i}', color = cmap((i)/len(R_block)))

    plt.title(f"Current distribution (Pocklington equation: {sum(len(element_currents[i]) for i in range(len(element_currents)))} elements)", size = 13)
    plt.ylabel("Induced current (mA)", size = 12)
    plt.xlabel("Z position (m)", size = 12)
    plt.grid(zorder = 0)
    plt.legend()
    
def plot_separately (R_block, current_block): 
    cmap = plt_cmaps['hot']
    cols = 3
    rows = (len(R_block)+cols-1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(10+3, rows * 3+3))
    axs = axs.flatten()
    for i in range(len(R_block)):
        axs[i].plot(R_block[i][:,2], np.abs(current_block[i])*1000, color = cmap((i)/len(R_block)))
        axs[i].set_title(f'Current distribution on {i+1} element', size = 6)
        axs[i].set_ylabel('Induced current (mA)', size = 6)
        axs[i].set_xlabel('Z position (m)', size = 6)
        axs[i].grid(zorder = 0)
        axs[i].legend()
    for j in range(len(R_block), len(axs)):
        fig.delaxes(axs[j])