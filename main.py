import argparse
import json
from evolve import run_simulation
from diagnostics import save_results, plot_diagnostics

def main():
    parser = argparse.ArgumentParser(description="Real-time wavefunction collapse simulation")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file", default=None)
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)

        x, psi_final, results = run_simulation(config)
        save_results(results, config)
        plot_diagnostics(results, x, config)

if __name__ == "__main__":
    main()
