# Problem 6(b) Monte Carlo estimate and convergence plot

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
analytic_p = 23 * np.pi / 192

# N values.
N_max = 1_000_000
X = rng.random(N_max)
Y = rng.random(N_max)
Z = rng.random(N_max)

# Conditions
success = (X**2 + Y**2 < Z) & (Z**2 > X * Y)
cumulative_successes = np.cumsum(success)

# Algorithm
N_values = np.unique(np.logspace(2, 6, 60).astype(int))
estimates = cumulative_successes[N_values - 1] / N_values

mc_p = success.mean()
mc_se = np.sqrt(mc_p * (1 - mc_p) / N_max)

# Aproximations 
print(f"Analytic probability   = {analytic_p:.6f}")
print(f"Monte Carlo probability= {mc_p:.6f}")
print(f"Monte Carlo SE         = {mc_se:.6f}")
print(f"Absolute error         = {abs(mc_p - analytic_p):.6f}")

# Plot
plt.figure(figsize=(8, 5))
plt.semilogx(N_values, estimates, label='Monte Carlo estimate')
plt.axhline(analytic_p, linestyle='--', label=fr'Analytic $23\pi/192={analytic_p:.4f}$')
plt.xlabel('Sample size N')
plt.ylabel('Estimated probability')
plt.title('Monte Carlo convergence for Problem 6')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()
