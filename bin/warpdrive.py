import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Const
c = 3e8
G = 6.67430e-11
hbar = 1.0545718e-34
k_B = 1.380649e-23
M_warp = 1e25  # hypothetical exotic matter

# Alcubierre metric parameters
R = 3  # Warp bubble radius
sigma = 8  # Controls bubble thickness
v_max = 2 * c  # Maximum warp velocity


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

    n_x, n_y = (warp_x, warp_y) / np.sqrt(warp_x**2 + warp_y**2)

    T_00 = -(c**4 / (8 * np.pi * G)) * ((df_dr)**2 + (sigma**2 / r**2) * (1 - f**2)**2)
    T_0x = -v * T_00 * n_x
    T_0y = -v * T_00 * n_y
    T_xx = T_00 * n_x**2
    T_yy = T_00 * n_y**2
    T_xy = T_00 * n_x * n_y
    
    return T_00, T_0x, T_0y, T_xx, T_yy, T_xy

def calculate_curvature(X, Y, v, warp_x, warp_y):
    T_00, T_0x, T_0y, T_xx, T_yy, T_xy = alcubierre_tensor(X, Y, v, warp_x, warp_y)
    curvature = (8 * np.pi * G / c**4) * (T_00 + T_xx + T_yy)
    return curvature

def hawking_temperature(v):
    # Simplified model: temperature proportional to acceleration
    a = v * c / R
    T = (a * hbar) / (2 * np.pi * c * k_B)
    return T

def causality_stress(X, Y, v, warp_x, warp_y):
    if v > c:
        r = np.sqrt((X - warp_x)**2 + (Y - warp_y)**2)
        # Increases as v exceeds c and as r decreases
        stress = (v/c - 1) * np.exp(-r)
        return stress
    return np.zeros_like(X)

def total_energy(Z, v):
    # if v >= c:
        #v = 0.999 * c
    E = -np.sum(Z[Z < 0]) * G / c**4
    E += 0.5 * M_warp * v**2 / np.sqrt(1 - (v/c)**2)
    return E

def relativistic_factor(velocity):
    #if velocity >= c:
     #   velocity = 1.5 * c  # Ensure velocity does not exceed the speed of light
    beta = velocity / c
    return 1 / np.sqrt(1 - beta**2)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


warp_x = warp_y = v = last_v = 0
dt = 0.1

def update_warp_bubble(frame):
    global warp_x, warp_y, v, last_v, dt

    warp_x = 5 * np.sin(frame / 50)
    warp_y = 5 * np.cos(frame / 50)
    v = v_max * np.sin(frame / 100)

    Z = calculate_curvature(X, Y, v, warp_x, warp_y)
    Z = np.clip(Z, -1, 1)

    C = causality_stress(X, Y, v, warp_x, warp_y)
    T_H = hawking_temperature(v)
    E_total = total_energy(Z, v)

    a = (v - last_v) / dt
    dt = min(0.1, 0.01 * c / max(abs(a), 1))
    last_v = v

    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, vmin=-1, vmax=1)
    ax.plot_surface(X, Y, C, cmap='hot', alpha=0.7)

    theta = np.linspace(0, 2*np.pi, 100)
    boundary_x = warp_x + R * np.cos(theta)
    boundary_y = warp_y + R * np.sin(theta)
    ax.plot(boundary_x, boundary_y, np.zeros(100), color='red', linewidth=3)

    gamma = relativistic_factor(v)
    ax.quiver(warp_x, warp_y, 0, v/v_max, 0, 0, color='blue', length=3, arrow_length_ratio=0.5)
    ax.text(warp_x, warp_y, 0, f'{v/c:.2f}c', color='red')
    ax.text(-10, -10, 0, f'E = {E_total:.2e} J', color='purple')
    ax.text(warp_x, warp_y, 1, f'T_H = {T_H:.2e} K', color='orange')
    ax.text(-10, 10, 0, f'γ = {gamma:.2f}', color='green')
    
    ax.set_zlim(-1, 1)
    ax.set_title(f'Alcubierre Warp Bubble - Frame {frame}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Space-Time Curvature / Time Dilation')

ani = animation.FuncAnimation(fig, update_warp_bubble, frames=200, interval=50)
ani.save('../data/warp_bubble_alcubierre.mp4', writer='ffmpeg')

# Optionally, show the plot (may be slow for high-res animations)
