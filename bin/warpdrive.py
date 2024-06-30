'''
WarDrive: Alcubierre Warp Drive Simulation
Mattia Braga
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Const
c = 3e8
G = 6.674e-11
hbar = 1.054571817e-34
k_B = 1.380649e-23
M = 1e25  # hypothetical exotic matter

# Alcubierre metric parameters
R = 3  # Warp bubble radius (m)
sigma = 8  # Controls bubble thickness
v_max = 10 * c  # Maximum warp velocity (10c)

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)


def shape_function(r, R, sigma):
    top = np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))
    bottom = 2 * np.tanh(sigma * R)
    return top / bottom


def alcubierre_tensor(X, Y, v, warp_x, warp_y):
    r = np.sqrt((X - warp_x)**2 + (Y - warp_y)**2)
    f = shape_function(r, R, sigma)
    df_dr = (sigma / 2) * (1 / np.cosh(sigma * (r - R))**2 - 1 / np.cosh(sigma * (r + R))**2) / np.tanh(sigma * R)
    
    # Unit vector in ship's direction
    n_x, n_y = (warp_x, warp_y) / np.sqrt(warp_x**2 + warp_y**2)
    
    # Tensor components (simplified)
    T_00 = -(c**4 / (8 * np.pi * G)) * ((df_dr)**2 + (sigma**2 / r**2) * (1 - f**2)**2)
    T_0x = -v * T_00 * n_x
    T_0y = -v * T_00 * n_y
    T_xx = T_00 * n_x**2
    T_yy = T_00 * n_y**2
    T_xy = T_00 * n_x * n_y
    
    return T_00, T_0x, T_0y, T_xx, T_yy, T_xy


def calculate_curvature(X, Y, v, warp_x, warp_y):
    T_00, T_0x, T_0y, T_xx, T_yy, T_xy = alcubierre_tensor(X, Y, v, warp_x, warp_y)
    # Einstein's equation: G_μν = (8πG/c⁴)T_μν
    curvature = (8 * np.pi * G / c**4) * (T_00 + T_xx + T_yy)  # Trace of tensor
    return np.asnumpy(curvature)


def alcubierre_tensor(X, Y, v, warp_x, warp_y):
    r = np.sqrt((X - warp_x)**2 + (Y - warp_y)**2)
    f = shape_function(r, R, sigma)
    df_dr = (sigma / 2) * (1 / np.cosh(sigma * (r - R))**2 - 1 / np.cosh(sigma * (r + R))**2) / np.tanh(sigma * R)
    
    # Unit vector in ship's direction
    n_x, n_y = (warp_x, warp_y) / np.sqrt(warp_x**2 + warp_y**2)
    
    # Tensor components (simplified)
    T_00 = -(c**4 / (8 * np.pi * G)) * ((df_dr)**2 + (sigma**2 / r**2) * (1 - f**2)**2)
    T_0x = -v * T_00 * n_x
    T_0y = -v * T_00 * n_y
    T_xx = T_00 * n_x**2
    T_yy = T_00 * n_y**2
    T_xy = T_00 * n_x * n_y
    
    return T_00, T_0x, T_0y, T_xx, T_yy, T_xy