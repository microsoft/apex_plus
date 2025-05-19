import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def alpha_beta_model(x, alpha, beta):
    """
    Collective communication time model:
    T = alpha + beta * x
    where:
      - alpha: latency (us)
      - beta: inverse bandwidth (us/KB)
      - x: message size (KB)
    """
    return alpha + beta * x

def fit_alpha_beta(x, y):
    """
    Fit the alpha-beta model to the data.
    Returns (alpha, beta)
    """
    # Initial guess: latency = min(y), bandwidth = (max(y)-min(y))/(max(x)-min(x))
    p0 = [np.min(y), (np.max(y) - np.min(y)) / (np.max(x) - np.min(x) + 1e-12)]
    # Only fit to positive x and y
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]
    popt, _ = curve_fit(alpha_beta_model, x_fit, y_fit, p0=p0, maxfev=10000)
    return popt

def plot_communication(filename, title, save_name):
    """Plot communication pattern and fit alpha-beta model."""
    # Read data
    data = pd.read_csv(filename)
    
    plt.figure(figsize=(10, 6))
    
    for n_gpus in sorted(data['num_gpus_per_node'].unique()):
        subset = data[data['num_gpus_per_node'] == n_gpus]
        x = subset['size(kb)'].values
        y = subset['time(us)'].values
        
        # Plot raw data
        plt.plot(x, y, marker='o', label=f'{n_gpus} GPUs (data)')
        
        # Fit alpha-beta model
        try:
            alpha, beta = fit_alpha_beta(x, y)
            x_fit = np.logspace(np.log10(np.min(x[x>0])), np.log10(np.max(x)), 100)
            y_fit = alpha_beta_model(x_fit, alpha, beta)
            plt.plot(x_fit, y_fit, '--', 
                     label=f'{n_gpus} GPUs (fit): $T=%.3e+%.3e x$' % (alpha, beta))
        except Exception as e:
            print(f"Curve fitting failed for {n_gpus} GPUs in {title}: {e}")

    plt.title(f'{title} Communication Time vs Size')
    plt.xlabel('Size (KB)')
    plt.ylabel('Time (μs)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig(f"{save_name}.png")
    plt.close()

def plot_all_patterns():
    """Plot all communication patterns"""
    patterns = [
        ("alltoall.csv", "All-to-All", "alltoall_fitted"),
        ("all_gather.csv", "All-Gather", "allgather_fitted"),
        ("all_reduce.csv", "All-Reduce", "allreduce_fitted"),
        ("reduce_scatter.csv", "Reduce-Scatter", "reducescatter_fitted")
    ]
    
    for csv_file, title, save_name in patterns:
        try:
            plot_communication(csv_file, title, save_name)
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

def main():
    plot_all_patterns()

if __name__ == "__main__":
    main()
