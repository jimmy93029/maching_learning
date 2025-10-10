import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian(mu, s, num_samples=1, n_uniform=840):
    """
    1. (a)
    Generate random samples from a Gaussian distribution with mean mu and variance s
    using the Central Limit Theorem.
    
    :param mu: Mean of the distribution
    :param s: Variance of the distribution
    :param num_samples: Number of samples to generate
    :param n_uniform: Number of uniform U(0,1) deviates to sum
    :return: Array of generated samples
    """
    # Generate the standard normal samples (mean 0, variance 1)
    uniform_samples = np.random.rand(num_samples, n_uniform)
    normal_samples = (np.sum(uniform_samples, axis=1) - n_uniform/2)/np.sqrt(n_uniform/12)  
    
    # Transform to N(mu, s)
    transformed_samples = mu + np.sqrt(s) * normal_samples
    
    return transformed_samples

def polynomial_basis_linear_model_data_generator(n, w, a, num_samples=1):
    """
    1. (b)
    Generate data points (x, y) from a polynomial basis linear model with noise.
    :param n: Number of basis functions
    :param w: Weight vector (length n)
    :param a: Variance of the noise
    :param num_samples: Number of samples to generate
    :return: Array of generated (x, y) points
    """
    # Generate uniform x values between -1 and 1
    x = np.random.uniform(-1, 1, num_samples)
    
    # Apply polynomial basis (phi(x) = [x^0, x^1, ..., x^(n-1)])
    phi_x = np.array([x**i for i in range(n)]).T
    
    # Calculate y = W^T * phi(x) + e (where e ~ N(0, a))
    e = generate_gaussian(0, a)
    y = np.dot(phi_x, w) + e  # Linear model with noise
    
    return x, y

def output():
    plt.figure(figsize=(8, 6))
    plt.hist(approx_normal_samples, bins=30, density=True, alpha=0.7, color='g')
    plt.title("Histogram of Approximate Normal Distribution (Central Limit Theorem Method)")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig("normal distribution")