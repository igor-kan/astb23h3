"""
ATB23: Lane-Emden Polytropic Stellar Structure Solver
Integrates the Lane-Emden equation for polytropic indices n = 0, 1, 1.5, 3, 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lane_emden_system(xi, y, n):
    """
    y = [theta, dtheta/dxi]
    Lane-Emden: d/dxi(xi^2 dtheta/dxi) + xi^2 theta^n = 0
    d^2theta/dxi^2 = -2/xi dtheta/dxi - theta^n
    """
    theta, dtheta = y
    if xi == 0:
        return [0.0, 0.0]
    
    # Avoid complex values if theta drops below zero
    theta_pos = max(theta, 0.0)
    d2theta = - (2.0 / xi) * dtheta - theta_pos**n
    return [dtheta, d2theta]

def solve_polytrope(n, xi_max=10.0):
    # Taylor expansion near xi = 0 to avoid 1/xi singularity:
    # theta(xi) ~ 1 - xi^2 / 6 + n xi^4 / 120
    xi0 = 1e-4
    theta0 = 1.0 - xi0**2 / 6.0
    dtheta0 = -xi0 / 3.0

    def event_surface(xi, y, n):
        return y[0]  # root when theta = 0 (stellar surface)
    event_surface.terminal = True
    event_surface.direction = -1

    sol = solve_ivp(lane_emden_system, [xi0, xi_max], [theta0, dtheta0],
                    args=(n,), events=event_surface, rtol=1e-8, atol=1e-8,
                    max_step=0.01)
    return sol.t, sol.y[0]

def plot_polytropes():
    plt.figure(figsize=(9, 5.5))
    indices = [0, 1, 1.5, 3.0]
    colors = ['#34495e', '#2980b9', '#27ae60', '#e74c3c']

    for n, col in zip(indices, colors):
        xi, theta = solve_polytrope(n)
        plt.plot(xi, theta, label=f'n = {n} (Surface $\\xi_1 = {xi[-1]:.2f}$)', color=col, lw=2)

    plt.axhline(0, color='black', linestyle=':', alpha=0.6)
    plt.xlim(0, 7.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'Dimensionless Radius $\xi = r / \alpha$', fontsize=11, fontweight='bold')
    plt.ylabel(r'Dimensionless Density Profile $\theta(\xi) = (\rho / \rho_c)^{1/n}$', fontsize=11, fontweight='bold')
    plt.title('Lane-Emden Polytropic Stellar Density Profiles', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('lane_emden_profiles.png', dpi=200)
    print("Polytrope plot saved as lane_emden_profiles.png")

if __name__ == '__main__':
    plot_polytropes()
