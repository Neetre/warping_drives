import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
c = 3e8                           # Speed of light [m/s]
G = 6.67430e-11                   # Gravitational constant [m^3/kg/s^2]
hbar = 1.0545718e-34              # Reduced Planck constant [J·s]
k_B = 1.380649e-23                # Boltzmann constant [J/K]
M_warp = 1e25                     # Hypothetical exotic matter mass [kg]

# Alcubierre Drive parameters
R = 3                             # Bubble radius
sigma = 8                         # Controls bubble thickness
v_max = 3 * c                     # Maximum (nominal) warp speed

# Create simulation grid
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

def shape_function(r, R, sigma):
    """Defines the smooth shape of the warp bubble."""
    return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (2 * np.tanh(sigma * R))

def alcubierre_tensor(X, Y, v, warp_x, warp_y, vx, vy):
    """
    Computes a simplified stress-energy distribution for the warp bubble.
    The calculation avoids division by zero by replacing r=0 with a small epsilon.
    """
    r = np.sqrt((X - warp_x)**2 + (Y - warp_y)**2)
    r_safe = np.where(r == 0, 1e-10, r)
    
    f = shape_function(r, R, sigma)
    df_dr = (sigma / 2) * (1/np.cosh(sigma*(r - R))**2 - 1/np.cosh(sigma*(r + R))**2) / np.tanh(sigma * R)
    
    T_00 = -(c**4 / (8 * np.pi * G)) * ((df_dr)**2 + (sigma**2/(r_safe**2))*(1 - f**2)**2)
    T_0x = -v * T_00 * vx
    T_0y = -v * T_00 * vy
    T_xx = T_00 * vx**2
    T_yy = T_00 * vy**2
    T_xy = T_00 * vx * vy
    return T_00, T_0x, T_0y, T_xx, T_yy, T_xy

def calculate_curvature(X, Y, v, warp_x, warp_y, vx, vy):
    """Computes a simplified measure of space-time curvature from the tensor components."""
    T_00, T_0x, T_0y, T_xx, T_yy, T_xy = alcubierre_tensor(X, Y, v, warp_x, warp_y, vx, vy)
    curvature = (8 * np.pi * G / c**4) * (T_00 + T_xx + T_yy)
    return curvature

def hawking_temperature(v):
    """
    Estimates a Hawking-like temperature proportional to an acceleration scale.
    Here we use a = |v| * c / R.
    """
    a = abs(v) * c / R
    T = (a * hbar) / (2 * np.pi * c * k_B)
    return T

def causality_stress(X, Y, v, warp_x, warp_y):
    """
    Computes an ad hoc 'causality stress' that turns on when the warp speed exceeds c.
    """
    if abs(v) > c:
        r = np.sqrt((X - warp_x)**2 + (Y - warp_y)**2)
        stress = (abs(v)/c - 1) * np.exp(-r)
        return stress
    else:
        return np.zeros_like(X)

def total_energy(Z, v):
    """
    Computes a total energy from the negative curvature contributions and a kinetic term.
    The speed is clamped when computing the relativistic kinetic energy.
    """
    E_curvature = -np.sum(Z[Z < 0]) * G / c**4
    v_eff = min(abs(v), 0.999 * c)
    gamma = 1 / np.sqrt(1 - (v_eff/c)**2)
    E_kinetic = 0.5 * M_warp * v_eff**2 * gamma
    return E_curvature + E_kinetic

def relativistic_factor(v):
    """Returns the Lorentz factor (γ), clamping the speed if needed."""
    v_eff = min(abs(v), 0.999 * c)
    beta = v_eff / c
    return 1 / np.sqrt(1 - beta**2)

# Set up the matplotlib 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Global state variables for the bubble’s position and speed
warp_x = 0.0
warp_y = 0.0
v = 0.0
last_v = 0.0
dt = 0.1

def update_warp_bubble(frame):
    global warp_x, warp_y, v, last_v, dt
    
    # Update the bubble's center along a circular trajectory
    new_warp_x = 5 * np.sin(frame / 50)
    new_warp_y = 5 * np.cos(frame / 50)
    
    # Compute the velocity direction from the change in position
    dx = new_warp_x - warp_x
    dy = new_warp_y - warp_y
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0:
        vx, vy = 1, 0
    else:
        vx, vy = dx / norm, dy / norm
    
    warp_x = new_warp_x
    warp_y = new_warp_y
    
    # Define the bubble's speed (this can be adjusted for different dynamics)
    v = v_max * np.sin(frame / 100)
    
    # Compute space-time curvature and related quantities
    Z = calculate_curvature(X, Y, v, warp_x, warp_y, vx, vy)
    Z = np.clip(Z, -1, 1)
    C = causality_stress(X, Y, v, warp_x, warp_y)
    T_H = hawking_temperature(v)
    E_total = total_energy(Z, v)
    
    # Update the time step dt based on a simple acceleration estimate
    a = (v - last_v) / dt
    dt = min(0.1, 0.01 * c / max(abs(a), 1))
    last_v = v

    # Clear and redraw the plot
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, vmin=-1, vmax=1)
    ax.plot_surface(X, Y, C, cmap='hot', alpha=0.7)
    
    # Draw the bubble's boundary
    theta = np.linspace(0, 2*np.pi, 100)
    boundary_x = warp_x + R * np.cos(theta)
    boundary_y = warp_y + R * np.sin(theta)
    ax.plot(boundary_x, boundary_y, np.zeros(100), color='red', linewidth=3)
    
    # Draw an arrow indicating the bubble's velocity direction
    gamma = relativistic_factor(v)
    ax.quiver(warp_x, warp_y, 0, vx, vy, 0, color='blue', length=3, arrow_length_ratio=0.5)
    
    # Annotate the plot with current simulation parameters
    ax.text(warp_x, warp_y, 0, f'{v/c:.2f}c', color='red')
    ax.text(-10, -10, 0, f'E = {E_total:.2e} J', color='purple')
    ax.text(warp_x, warp_y, 1, f'T_H = {T_H:.2e} K', color='orange')
    ax.text(-10, 10, 0, f'γ = {gamma:.2f}', color='green')
    
    ax.set_zlim(-1, 1)
    ax.set_title(f'Alcubierre Warp Bubble - Frame {frame}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Curvature / Stress')

ani = animation.FuncAnimation(fig, update_warp_bubble, frames=200, interval=50)
ani.save('warp_bubble_alcubierre.mp4', writer='ffmpeg')
