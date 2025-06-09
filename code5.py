#!/usr/bin/env python3
"""
✨ The Cosmic Meta-Optimizer: Searching for the Genesis Parameter ✨

This script performs a meta-optimization to answer the question:
"What initial condition (`BASE_BETA`) for the Noise Sea is most likely to
produce a universe capable of high, stable complexity?"

It systematically sweeps through different `BASE_BETA` values. For each one,
it generates a full 2D phase diagram of universal complexity and calculates
a "fitness score" based on the size of the "Goldilocks Zone"
(the region where equilibrium β is in a life-friendly range).

The final plot of "Fitness vs. BASE_BETA" reveals the optimal "Genesis Parameter"
for creating an interesting universe.

Usage:
    python cosmic_meta_optimizer.py

Dependencies:
    numpy, matplotlib, scipy, tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import multiprocessing
from tqdm import tqdm
import time

# --- Core Simulation & Measurement Functions ---

def generate_fractal_noise(N, beta=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(N).reshape(N, 1)
    ky = np.fft.fftfreq(N).reshape(1, N)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    amplitude = 1.0 / (k2 ** (beta / 2.0))
    phases = np.exp(2j * np.pi * np.random.rand(N, N))
    spectrum = amplitude * phases
    noise = np.fft.ifft2(spectrum).real
    return (noise - noise.mean()) / (noise.std() + 1e-12)

def estimate_global_beta(grid):
    N = grid.shape[0]
    fft_grid = np.fft.fftshift(np.fft.fft2(grid))
    power_spectrum = np.abs(fft_grid)**2
    center_x, center_y = N // 2, N // 2
    y, x = np.indices((N, N))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    power_sum_per_bin = np.bincount(r.ravel(), power_spectrum.ravel())
    count_per_bin = np.bincount(r.ravel())
    valid_bins = count_per_bin > 0
    power_profile = power_sum_per_bin[valid_bins] / count_per_bin[valid_bins]
    freqs = np.arange(len(count_per_bin))[valid_bins]
    valid_points = (freqs > 1) & (freqs < N // 4) & (power_profile > 1e-9)
    if np.sum(valid_points) < 5: return np.nan
    log_freqs = np.log10(freqs[valid_points])
    log_power = np.log10(power_profile[valid_points])
    try:
        slope, _ = np.polyfit(log_freqs, log_power, 1)
        return -slope
    except np.linalg.LinAlgError:
        return np.nan

class FractalAmplifierCA:
    def __init__(self, **params):
        self.params = params
        self.grid_size = params['grid_size']
        self.grid = generate_fractal_noise(self.grid_size, beta=params['base_beta'])
        self.amplifiers = np.random.choice(self.grid_size**2, params['num_amplifiers'], replace=False)
        self.amplifier_rows, self.amplifier_cols = np.unravel_index(self.amplifiers, (self.grid_size, self.grid_size))
        self.amplified_patch = generate_fractal_noise(32, beta=params['amplified_beta']) * params['amplification_strength']
        self.diffusion_kernel = np.array([[0.5, 1, 0.5], [1, -6, 1], [0.5, 1, 0.5]]) * params['diffusion_rate']
        self.complexity_threshold = params['complexity_threshold']

    def step(self):
        self.grid += convolve(self.grid, self.diffusion_kernel, mode='wrap')
        for r, c in zip(self.amplifier_rows, self.amplifier_cols):
            patch_radius = 5
            patch = self.grid[max(0, r-patch_radius):min(self.grid_size, r+patch_radius+1),
                              max(0, c-patch_radius):min(self.grid_size, c+patch_radius+1)]
            if patch.std() > self.complexity_threshold:
                patch_size = self.amplified_patch.shape[0]
                row_start, col_start = r - patch_size//2, c - patch_size//2
                if row_start >= 0 and row_start+patch_size <= self.grid_size and col_start >= 0 and col_start+patch_size <= self.grid_size:
                    self.grid[row_start:row_start+patch_size, col_start:col_start+patch_size] += self.amplified_patch
        self.grid = (self.grid - self.grid.mean()) / (self.grid.std() + 1e-12)

    def run_to_equilibrium(self):
        beta_history = []
        for i in range(self.params['num_steps']):
            self.step()
            if i >= self.params['num_steps'] - self.params['equilibrium_window']:
                current_beta = estimate_global_beta(self.grid)
                if not np.isnan(current_beta):
                    beta_history.append(current_beta)
        return np.mean(beta_history) if beta_history else np.nan

# --- Worker Function for a Single Phase Diagram ---

def generate_phase_diagram(base_beta_value):
    """
    Generates a full phase diagram for a given BASE_BETA and returns
    its fitness score (size of the Goldilocks Zone).
    """
    RESOLUTION = 10 # Lower resolution for faster meta-sweep
    diffusion_rates = np.linspace(0.01, 0.3, RESOLUTION)
    amplification_strengths = np.linspace(0.01, 0.3, RESOLUTION)

    SIM_PARAMS = {
        'grid_size': 128, # Smaller grid for speed
        'num_amplifiers': 15,
        'base_beta': base_beta_value,
        'amplified_beta': base_beta_value + 1.2, # Amplified is relative to base
        'complexity_threshold': 0.1,
        'num_steps': 300, # Fewer steps for faster equilibrium
        'equilibrium_window': 50
    }

    beta_grid = np.zeros((RESOLUTION, RESOLUTION))
    for i, strength in enumerate(amplification_strengths):
        for j, diffusion in enumerate(diffusion_rates):
            sim = FractalAmplifierCA(
                diffusion_rate=diffusion,
                amplification_strength=strength,
                **SIM_PARAMS
            )
            final_beta = sim.run_to_equilibrium()
            beta_grid[i, j] = final_beta if not np.isnan(final_beta) else 0

    # Define Goldilocks Zone (e.g., equilibrium beta between 3.5 and 5.0)
    goldilocks_min_beta = 3.5
    goldilocks_max_beta = 5.0
    goldilocks_zone_mask = (beta_grid >= goldilocks_min_beta) & (beta_grid <= goldilocks_max_beta)
    fitness_score = np.sum(goldilocks_zone_mask)
    
    print(f"Finished for BASE_BETA={base_beta_value:.2f}, Fitness Score={fitness_score}")
    return base_beta_value, fitness_score

# --- Main Meta-Optimizer Orchestrator ---

if __name__ == '__main__':
    # Define the range of initial conditions (BASE_BETA) to test
    base_beta_sweep = np.linspace(0.5, 4.0, 15)

    print("🚀 LAUNCHING COSMIC META-OPTIMIZER 🚀")
    print(f"Searching for the optimal initial condition (BASE_BETA) over {len(base_beta_sweep)} universes.")
    print("This will take a significant amount of time...")
    
    start_time = time.time()

    # We will run this sequentially for clarity, but it can be parallelized
    fitness_results = []
    for base_beta in tqdm(base_beta_sweep):
        _, fitness = generate_phase_diagram(base_beta)
        fitness_results.append(fitness)

    end_time = time.time()
    print(f"\nMeta-optimization complete in {end_time - start_time:.2f} seconds.")

    # --- Plot the Final Result: The Genesis Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(base_beta_sweep, fitness_results, 'o-', color='purple', linewidth=2, markersize=8)
    
    # Find and highlight the peak
    if fitness_results:
        max_fitness_idx = np.argmax(fitness_results)
        optimal_base_beta = base_beta_sweep[max_fitness_idx]
        max_fitness = fitness_results[max_fitness_idx]
        ax.axvline(optimal_base_beta, color='gold', linestyle='--', label=f'Optimal Genesis β ≈ {optimal_base_beta:.2f}')
        ax.scatter(optimal_base_beta, max_fitness, color='gold', s=150, zorder=5, marker='*')
        print(f"\n🏆 Optimal Genesis Parameter Found: BASE_BETA ≈ {optimal_base_beta:.2f}")

    ax.set_title("The Genesis Curve: Optimal Initial Condition for a Complex Universe", fontsize=16)
    ax.set_xlabel("Initial Fractal Complexity of the Universe (BASE_BETA)", fontsize=12)
    ax.set_ylabel("Fitness (Size of Goldilocks Zone)", fontsize=12)
    ax.grid(True, linestyle=':')
    ax.legend()
    plt.show()