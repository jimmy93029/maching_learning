import numpy as np
from utils import generate_gaussian  # Assuming this is the Gaussian generator function

# Sequential estimator based on Welford's algorithm
def sequential_estimator(mu, s, num_samples=1000):
    """
    Sequentially estimate the mean and variance of the data generated from N(mu, s)
    using Welford's online algorithm.

    :param mu: Mean of the distribution
    :param s: Variance of the distribution
    :param num_samples: Number of samples to generate
    :param n_uniform: Number of uniform U(0,1) deviates to sum
    :return: The final estimated mean and variance
    """
    # Initial estimates
    mu_n = 0  # Initial mean estimate
    sigma_n = 0  # Initial variance estimate
    sample_count = 0  # Number of samples processed
    
    # Sequential estimation
    for i in range(1, num_samples + 1):
        # Generate a new sample from the Gaussian distribution
        x_n = generate_gaussian(mu, s)
        
        # Increment sample count
        sample_count += 1
        
        # Update mean using Welford's formula
        mu_n_sub_1 = mu_n  # Previous mean
        mu_n = (mu_n * (sample_count - 1) + x_n) / sample_count
        
        # Update variance using Welford's formula
        sigma_n_sub_1 = sigma_n  # Previous variance
        sigma_n = (sigma_n_sub_1 * (sample_count - 1) + (x_n - mu_n) * (x_n - mu_n_sub_1)) / sample_count
        
        # Optionally: print the current mean and variance
        print(f"Add data point: {x_n[0]}")  # Access the first value of the array
        print(f"Mean = {mu_n[0]:.4f}    Variance = {sigma_n[0]:.4f}")  # Ensure scalar output
    
    return mu_n, sigma_n

# Example usage
mu = 3  # Desired mean
s = 5   # Desired variance
estimated_mean, estimated_variance = sequential_estimator(mu, s, num_samples=1000)

# Final estimated mean and variance
print(f"Final Estimated Mean: {estimated_mean[0]:.4f}")
print(f"Final Estimated Variance: {estimated_variance[0]:.4f}")
