
import numpy as np
from geometry import calculate_positions
from matrix_elements_Hallen import calculate_impedance_Hallen
from matrix_elements_Hallen import calculate_solution_Hallen

#* Define constants
light_speed, mu0, eps0, frequency = 299792458., 4*np.pi*1e-7, 8.854e-12, 1e6 * 148 #* Also define an operating frequency 

#* Calculate some parametres of system
wavelength, wavenumber = light_speed / frequency, 2 * np.pi * frequency / light_speed

#* Space resolution of system
epsilon = 1000

#* set geometry of antenna
source_position = np.array([[0,0,0]])
element_position = np.array([0.0])
element_radii = np.array([5e-3])
element_length = np.array([2])

element_num, R = calculate_positions(element_length, element_position, frequency, epsilon)
impedance_block = calculate_impedance_Hallen(R, element_num, element_radii, wavenumber, wavelength, epsilon)
homogeneous, nonhomogeneous = calculate_solution_Hallen(R, element_num, wavenumber, source_position)
 