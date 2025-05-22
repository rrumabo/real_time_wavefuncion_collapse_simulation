import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def save_results(results, config):
    output_dir = config.get('output_dir', 'simulation_output')
    os.makedirs(output_dir, exist_ok=True)

    for key, value in results.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)

    energy_dissipation_rate = np.gradient(results['energy_history'], config['d_tau'])
    df = pd.DataFrame({
        "Tau": results['tau'],
        "Energy": results['energy_history'],
        "Energy Dissipation Rate": energy_dissipation_rate
    })
    df.to_csv(os.path.join(output_dir, "energy_dissipation_rate.csv"), index=False)

def plot_diagnostics(results, x, config):
    tau = results['tau']
    output_dir = config.get('output_dir', 'simulation_output')

    plt.figure()
    plt.plot(tau, results['entropy_history'], label="Entropy", color="blue")
    plt.plot(tau, results['energy_history'], label="Energy", color="red")
    plt.plot(tau, results['free_energy_history'], label="Free Energy", color="green")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Entropy, Energy, and Free Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "entropy_energy_real_time.png"))
    plt.close()
