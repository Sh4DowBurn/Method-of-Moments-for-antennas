import matplotlib.pyplot as plt
import numpy as np
import colormaps as cmaps
from matplotlib import colormaps as plt_cmaps
import plotly.graph_objects as go

light_speed, mu0, eps0 = 299792458., 4*np.pi*1e-7, 8.854e-12

def calculate_positions(antenna, delta_r):
    element_num  = np.zeros(len(antenna)-1, dtype = int)
    for i in range (len(antenna)-1):
        length = np.linalg.norm(antenna[i+1] - antenna[i])
        element_num[i] = int(length/delta_r) if int(length/delta_r)%2!=0 else int(length/delta_r)+1
    
    R_block = [0] * len(element_num)
    for m in range (0, len(element_num)):
        R_block[m] = np.zeros((element_num[m], 3))
        tau = (antenna[m+1] - antenna[m]) / np.linalg.norm(antenna[m+1] - antenna[m])
        for i in range (0, len(R_block[m])):
            R_block[m][i] = (antenna[m] + antenna[m+1]) / 2 - tau * delta_r * element_num[m] / 2 + tau * delta_r * (1/2 + i)
    
    num_elements = sum(element_num)
    R = np.zeros((sum(element_num),3))
    cum_n = np.append(0, np.cumsum(element_num))
    for i in range (len(cum_n) - 1):
        R[cum_n[i]:cum_n[i+1]] = R_block[i]
    return R_block, R
    
def draw_antenna_3d(R, source_position):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=R[:, 0],  
        y=R[:, 1],  
        z=R[:, 2],  
        mode='markers',
        marker=dict(color='blue', size=2), 
        name='Antenna'
    ))
    fig.add_trace(go.Scatter3d(
        x=source_position[:, 0],  
        y=source_position[:, 1], 
        z=source_position[:, 2],  
        mode='markers',
        marker=dict(color='red', size=2), 
        name='Source'
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Antenna visualization",
        showlegend=True
    )
    fig.show()


def current_distribution_3d (R, current):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=R[:,0],
        y=R[:,1],
        z=R[:,2],
        mode='markers',
        marker=dict(
            size=3,
            color=np.abs(current),
            colorscale='plasma',
            showscale=True,
            colorbar=dict(title='Amplitude of current, A')
        )
    ))
    fig.update_layout(
        title='Induced Current Distribution in 3D',
        scene=dict(
            xaxis_title='X position, m',
            yaxis_title='Y position, m',
            zaxis_title='Z position, m'
        )
    )
    fig.show()