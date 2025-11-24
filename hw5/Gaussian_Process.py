import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, det, LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import os


def rational_quadratic_kernel(A, B, params):
    """
    Computes the Rational Quadratic (RQ) kernel matrix between two input matrices A and B.
    """
    # Convert log parameters back to original scale
    sigma_f_sq = params[0]
    l = params[1]
    alpha = params[2]
    
    # Calculate the squared Euclidean distance matrix
    dist_sq = cdist(A, B, metric='sqeuclidean')
    
    # Apply the Rational Quadratic kernel formula: K = sigma_f^2 * (1 + dist^2 / (2 * alpha * l^2))^(-alpha)
    K = sigma_f_sq * (1.0 + dist_sq / (2.0 * alpha * l**2))**(-alpha)
    return K


def negative_log_likelihood(params, X_train, y_train, beta_inv):
    """
    Calculates the Negative Marginal Log-Likelihood (NMLL) for the GP model 
    """
    N = len(X_train)
    
    # Compute the kernel matrix K(X, X)
    K = rational_quadratic_kernel(X_train, X_train, params)
    
    # Construct the covariance matrix C = K + beta^-1 * I
    C = K + beta_inv * np.eye(N)
    
    try:
        C_inv = inv(C) 
        
        alpha = C_inv @ y_train
        
        quadratic_term = y_train.T @ alpha
        
        log_det_C = np.log(det(C))
        
        # MLL = - 0.5 * log(det(C)) - 0.5 * y.T @ C_inv @ y - N/2 * log(2 * pi)
        log_likelihood = -0.5 * log_det_C - 0.5 * quadratic_term - N/2 * np.log(2.0 * np.pi)
        
        # Return the Negative MLL
        return -log_likelihood.flatten()[0]
        
    except LinAlgError:
        # This error can occur if C is singular or ill-conditioned
        return 1e10


def gp_predict(X_train, y_train, X_test, kernel_params, beta_inv):
    """
    Performs Gaussian Process prediction on test points, returning the mean and variance.
    """
    N = len(X_train)
    # Compute the covariance matrix C and solve for alpha = C^-1 y
    K = rational_quadratic_kernel(X_train, X_train, kernel_params)
    C = K + beta_inv * np.eye(N)
    alpha = np.linalg.solve(C, y_train) 
    
    mu_star = []
    sigma_f_sq_star = []
    
    # Predict mean and variance for each test point x*
    for x_star in X_test:
        x_star = x_star.reshape(1, -1)
        # k* = K(X_train, x*)
        k_star = rational_quadratic_kernel(X_train, x_star, kernel_params)
        
        # Predicted Mean: mu(x*) = k*^T * alpha
        mu = k_star.T @ alpha
        mu_star.append(mu.flatten()[0])
        
        # Predicted Variance: sigma_f^2(x*) = k(x*, x*) - k*^T C^-1 k*
        k_star_star = rational_quadratic_kernel(x_star, x_star, kernel_params) + beta_inv * np.eye(N)
        v = np.linalg.solve(C, k_star) # v = C^-1 k*
        sigma_sq = k_star_star - k_star.T @ v
        sigma_f_sq_star.append(sigma_sq.flatten()[0])
        
    return np.array(mu_star), np.array(sigma_f_sq_star)


def optimize_gp_params(X, y, beta_inv):
    """
    Minimizes the NMLL using scipy.optimize.minimize to find the optimal kernel hyperparameters.
    """
    initial_params = np.array([1.0, 1.0, 1.0])
    
    # Set boundaries for parameters
    bounds = [(0.1, 50.0), (0.1, 50.0), (0.1, 50.0)]
    
    # Run the L-BFGS-B optimization algorithm
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_params,
        args=(X, y, beta_inv),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # Convert optimized parameters back to original scale
    optimized_params = result.x
    
    # Return results
    return optimized_params, result.fun, result.success


def draw(X, y, X_test, mu_star_opt, sigma_sq_star_opt, title_name):
    """
    Visualizes the GP prediction results, including the mean and 95% confidence interval.
    """
    # Calculate 95% confidence interval bounds (1.96 standard deviations)
    std_dev_opt = np.sqrt(sigma_sq_star_opt)
    lower_bound_opt = mu_star_opt - 1.96 * std_dev_opt
    upper_bound_opt = mu_star_opt + 1.96 * std_dev_opt

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Training Data (X, Y)', color='red', marker='x')
    plt.plot(X_test, mu_star_opt, label='Prediction Mean', color='blue')
    
    # Shade the 95% confidence region
    plt.fill_between(
        X_test.flatten(),
        lower_bound_opt,
        upper_bound_opt,
        alpha=0.2,
        color='lightblue',
        label='95% Confidence Interval'
    )
    
    plt.xlabel('X Input')
    plt.ylabel('f(X) / Y')
    plt.title(title_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(title_name)


def main():
    """
    Main function: loads data, optimizes parameters, performs prediction, and visualizes results.
    """
    # --- 5.1 Data Loading and Preparation ---
    if not os.path.exists('input.data'):
        print("Error: 'input.data' file not found.")
        return
        
    # Load data and reshape X and y to (N, 1) matrices
    data = np.loadtxt('input.data')
    X = data[:, 0].reshape(-1, 1)  # Training input X
    y = data[:, 1].reshape(-1, 1)  # Training observations y
    
    # Set noise parameter: beta = 5.0
    beta = 5.0
    beta_inv = 1.0 / beta
    
    # Generate test points X_test
    X_test = np.linspace(-60, 60, 200).reshape(-1, 1)

    # --- Task 1: Prediction with Initial Parameters ---
    initial_params = np.array([1.0, 1.0, 1.0])

    mu_star_init, sigma_sq_star_init = gp_predict(X, y, X_test, initial_params, beta_inv)

    draw(X, y, X_test, mu_star_init, sigma_sq_star_init, 'Gaussian Process Regression with Initial Rational Quadratic Kernel')

    # --- Task 2: Parameter Optimization ---
    
    # Optimize kernel hyperparameters
    optimized_params, nmll_value, success = optimize_gp_params(X, y, beta_inv)
    
    print("--- Gaussian Process Task 2: Optimization Results ---")
    print(f"Optimization Success: {success}")
    print(f"Optimized Parameters (sigma_f^2, l, alpha): {optimized_params}")
    print(f"Final Minimum NMLL: {nmll_value:.4f}")

    # --- Task 2: Prediction with Optimized Parameters ---
    
    # Use optimized parameters for GP prediction
    mu_star_opt, sigma_sq_star_opt = gp_predict(X, y, X_test, optimized_params, beta_inv)

    draw(X, y, X_test, mu_star_opt, sigma_sq_star_opt, 'Gaussian Process Regression with Optimized Rational Quadratic Kernel')


if __name__ == "__main__":
    main()