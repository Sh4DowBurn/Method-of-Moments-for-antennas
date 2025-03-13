import matplotlib.pyplot as plt
import numpy as np
import colormaps as cmaps
from matplotlib import colormaps as plt_cmaps
import plotly.graph_objects as go

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_positions(antenna, delta_r):
    element_num  = np.zeros(len(antenna.length), dtype = int)
    for i in range (len(antenna.length)):
        element_num[i] = int(antenna.length[i]/delta_r) if int(antenna.length[i]/delta_r)%2!=0 else int(antenna.length[i]/delta_r)+1
    
    R_block = [0] * len(element_num)
    for m in range (0, len(element_num)):
        R_block[m] = np.zeros((element_num[m], 2))
        delta_x = delta_r * np.cos(antenna.angle[m])
        delta_y = delta_r * np.sin(antenna.angle[m])
        for i in range (0, len(R_block[m])):
            R_block[m][i, 0] = antenna.position[m, 0] - element_num[m]*delta_x/2 + delta_x * (1/2 + i)
            R_block[m][i, 1] =  antenna.position[m, 1] - element_num[m]*delta_y/2 + delta_y * (1/2 + i)
    
    num_elements = sum(element_num)
    R = np.zeros((sum(element_num),2))
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n) - 1):
        R[cum_n[i]:cum_n[i+1]] = R_block[i]
    return R_block, R

def plot_antenna (R_block, R, antenna, delta_r):
    
    fig, ax = plt.subplots()
    for i in range (len(R_block)):
        delta_x = delta_r * np.sin(antenna.angle[i])
        delta_y = delta_r * np.cos(antenna.angle[i])
        plt.plot(np.array(R_block[i][:,0]),np.array(R_block[i][:,1]), linestyle = "--", marker = '.', markersize = '2', zorder = 10, color = 'blue')
    
    for m in range (len(R_block)):
        for i in range(len(R_block[m])):
            for k in range(len(antenna.source_position)):
                if all(antenna.source_position[k] == R_block[m][i]):
                    delta_x = delta_r * np.sin(antenna.angle[m])
                    delta_y = delta_r * np.cos(antenna.angle[m])
                    plt.errorbar(np.array(R_block[m][i,0]),np.array(R_block[m][i,1]), linestyle = "none", marker = '.', markersize = '5', zorder = np.inf, color = 'red')

    ax.set_xlim(min(R[:,0])-0.1, max(R[:,0])+0.1)
    ax.set_ylim(min(R[:,1])-0.1, max(R[:,1]+0.1))

    ax.axis('equal')
    ax.set_title('2D Model of antenna', size = 14)
    ax.set_xlabel('X position, m', size = 12)
    ax.set_ylabel('Y position, m', size = 12)

    ax.grid(True)

    plt.show()

def current_disribution_together (R_block, element_currents) :
    cmap = plt_cmaps['hot']
    for i in range (len(R_block)):
        plt.plot(R_block[i][:,1], np.abs(element_currents[i]*1000), zorder = np.inf, label = f'element {1+i}', color = cmap((i)/len(R_block)))

    plt.title(f"Current distribution ({sum(len(element_currents[i]) for i in range(len(element_currents)))} elements)", size = 13)
    plt.ylabel("Induced current (mA)", size = 12)
    plt.xlabel("Y position (m)", size = 12)
    plt.grid(zorder = 0)
    plt.legend()
    
def current_distribution_2d (R, current):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=R[:,0],
        y=R[:,1] ,
        mode='markers',
        marker=dict(
            size=10,
            color=np.abs(current),
            colorscale='plasma',
            showscale=True,
            colorbar=dict(title='Amplitude of current, A')
        ),
        name='Точки'
    ))
    fig.update_layout(
        title='Induced current distribution',
        xaxis_title='Y position, m',
        yaxis_title='Z position, m',
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(scaleanchor="x")
    )
    fig.show()