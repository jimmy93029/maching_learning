import numpy as np
import matplotlib.pyplot as plt
from utils import generate_gaussian

# --- 1. Generate synthetic polynomial regression data ---
def polynomial_basis_linear_model_data_generator(n, w, a, num_samples):
    x_values = np.random.uniform(-1.0, 1.0, num_samples)
    Poly_x = np.zeros((num_samples, n))

    # Build design matrix [1, x, x^2, ..., x^(n-1)]
    for i in range(n):
        Poly_x[:, i] = x_values**i

    # Gaussian observation noise
    e = generate_gaussian(0, a, num_samples=num_samples).reshape(-1, 1) 
    y = Poly_x @ w + e
    return Poly_x, y

# --- 2. Bayesian posterior update (online, per single point) ---
def update_posterior(X, y, mu_prev, Sigma_prev, a):
    beta = 1.0 / a  # Precision = 1 / variance
    y_matrix = np.atleast_2d(y)

    # Posterior covariance: (Sigma^-1 + beta * X^T X)^-1
    Sigma_inv = np.linalg.inv(Sigma_prev) + beta * X.T @ X
    Sigma_new = np.linalg.inv(Sigma_inv)

    # Posterior mean: Sigma_new (Sigma^-1 mu + beta * X^T y)
    mu_new = Sigma_new @ (np.linalg.inv(Sigma_prev) @ mu_prev + beta * X.T @ y_matrix)

    return mu_new, Sigma_new

# --- 3. Predictive distribution for a new point ---
def posterior_predictive(X, mu, Sigma, a):
    mean = (X @ mu).item(0)
    var = (X @ Sigma @ X.T + a).item(0)
    return mean, var

# --- 4. Output function to print and save logs in the requested format ---
def print_output(X_batch, Y_batch, mu_n, Sigma_n, predictive_mean, predictive_variance, filename="output_logs.txt"):
    # X_batch contains the data point, we log the actual value
    x_data = X_batch.flatten()
    y_data = Y_batch.flatten()

    with open(filename, 'a') as log_file:
        log_file.write(f"Add data point: ({x_data[1]}, {y_data}):\n")
        
        # Print posterior mean
        log_file.write("Posterior mean:\n")
        for i in range(len(mu_n)):
            log_file.write(f"  {mu_n[i][0]}\n")
        
        # Print posterior variance (Sigma_n)
        log_file.write("Posterior variance:\n")
        for row in Sigma_n:
            log_file.write("  " + ", ".join([f"{value:.8f}" for value in row]) + "\n")
        
        # Print predictive distribution mean and variance
        log_file.write(f"Predictive distribution ~ N({predictive_mean:.5f}, {predictive_variance:.5f})\n\n")


# --- 5. Plot results: Ground Truth, 10 pts, 50 pts, Final ---
def final_plots(plot_data, n, a, w, name):
    x_test = np.linspace(-2.0, 2.0, 200)
    X_test_design = np.array([x_test**i for i in range(n)]).T
    y_true_flat = (X_test_design @ w).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    plot_items = [
        {'mu': w.copy(), 'Sigma': np.zeros((n, n)), 'X_seen': np.empty((0, n)), 'y_seen': np.empty(0), 'title': 'Ground truth', 'pos': (0, 0)},
        {'mu': plot_data['final']['mu'], 'Sigma': plot_data['final']['Sigma'], 'X_seen': plot_data['final']['X_seen'], 'y_seen': plot_data['final']['y_seen'], 'title': 'Predict result', 'pos': (0, 1)},
        {'mu': plot_data['10']['mu'], 'Sigma': plot_data['10']['Sigma'], 'X_seen': plot_data['10']['X_seen'], 'y_seen': plot_data['10']['y_seen'], 'title': 'After 10 incomes', 'pos': (1, 0)},
        {'mu': plot_data['50']['mu'], 'Sigma': plot_data['50']['Sigma'], 'X_seen': plot_data['50']['X_seen'], 'y_seen': plot_data['50']['y_seen'], 'title': 'After 50 incomes', 'pos': (1, 1)},
    ]

    for item in plot_items:
        ax = axes[item['pos']]
        mu, Sigma = item['mu'], item['Sigma']
        X_seen, y_seen = item['X_seen'], item['y_seen']

        # Predictive mean and std
        y_mean = (X_test_design @ mu)
        y_var = (X_test_design @ Sigma @ X_test_design.T + a)

        ax.plot(x_test, y_mean, color='black', label="Mean")
        ax.plot(x_test, y_mean + y_var, color='red', linewidth=1)
        ax.plot(x_test, y_mean - y_var, color='red', linewidth=1)

        if item['title'] != 'Ground truth' and len(X_seen) > 0:
            ax.scatter(X_seen[:, 1], y_seen, color='blue', s=10)

        ax.set_title(item['title'])
        ax.set_xlim([-2, 2])
        ax.set_ylim([-20, 20])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{name}.png")


def Case(b, n, a, w, num_samples, name):
    X_full, y_full = polynomial_basis_linear_model_data_generator(n, w, a, 100)

    mu = np.zeros((n, 1))
    Sigma = np.eye(n) / b

    plot_data = {}

    for i in range(len(X_full)):
        X_batch = X_full[i].reshape(1, -1)
        Y_batch = y_full[i]

        mu, Sigma = update_posterior(X_batch, Y_batch, mu, Sigma, a)
        predictive_mean, predictive_variance = posterior_predictive(X_batch, mu, Sigma, a)
        print_output(X_batch, Y_batch, mu, Sigma, predictive_mean, predictive_variance, filename=f"{name}_logs.txt")

        current_count = i + 1
        if current_count in [10, 50]:
            plot_data[str(current_count)] = {'mu': mu.copy(), 'Sigma': Sigma.copy(), 'X_seen': X_full[:current_count], 'y_seen': y_full[:current_count]}

    plot_data['final'] = {'mu': mu.copy(), 'Sigma': Sigma.copy(), 'X_seen': X_full, 'y_seen': y_full}
    final_plots(plot_data, n, a, w, name)


# --- 5. Main execution ---
if __name__ == '__main__':
    name = "Case1"
    b = 1
    n = 4
    a = 1
    w = np.array([1, 2, 3, 4]).reshape(-1, 1)
    num_samples = 100

    Case(b, n, a, w, num_samples, name)


