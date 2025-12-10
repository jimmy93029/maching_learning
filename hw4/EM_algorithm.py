import numpy as np
import struct
import matplotlib.pyplot as plt

# --- 1. MNIST DATA LOADING FUNCTIONS (PARSING FUNCTIONS) ---

def load_mnist_images(filename):
    """
    Parses the MNIST image file (.idx3-ubyte)
    """
    with open(filename, 'rb') as f:
        # Read Magic Number and basic information
        # '>IIII' stands for 4 unsigned 4-byte integers (Big-Endian format)
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        # Read image data
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        
        # Reshape to (num_images, rows * cols)
        images = images.reshape(num_images, rows * cols)
        
    return images, rows, cols

def load_mnist_labels(filename):
    """
    Parses the MNIST label file (.idx1-ubyte)
    """
    with open(filename, 'rb') as f:
        # Read Magic Number and number of images
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        # Read label data
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
    return labels

# --- 2. DATA PREPROCESSING (Binarization) ---

def preprocess_mnist_data(images):
    """
    Binarizes the grayscale images (Binning into two bins)
    """
    # Grayscale values range 0-255. Use 127 as the threshold to split into two classes (0 and 1)
    # 0 (Background) -> 0, >127 (Stroke) -> 1
    binary_images = (images > 127).astype(np.float64) 
    return binary_images

# --- 3. EM ALGORITHM CORE UTILITIES ---

def log_sum_exp(log_probs, axis=None):
    """Log-Sum-Exp trick for numerical stability."""
    max_log_prob = np.max(log_probs, axis=axis, keepdims=True)
    # Avoid log(0)
    max_log_prob[max_log_prob == -np.inf] = 0 
    
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob), axis=axis, keepdims=True))

# --- 4. BERNOULLI EM CLUSTERING FUNCTION ---

def bernoulli_em_clustering(X, K, max_iter=100, tol=1e-5):
    """
    EM Algorithm for Bernoulli Mixture Model.
    """
    num_samples, num_features = X.shape
    np.random.seed(42)

    # Set a small numerical stability boundary (Epsilon)
    STABILITY_EPS = 1e-10
    
    # --- Initialize pi (Mixing Weights) ---
    pi = np.full(K, 1 / K)
    
    # --- Initialize mu (Bernoulli Parameters, breaking symmetry) ---
    
    # 1. Randomly select indices of K samples as initial centers
    random_indices = np.random.choice(num_samples, K, replace=False)
    # 2. Use the pixel values (0 or 1) of the selected K samples as initial mu
    mu = X[random_indices, :].astype(np.float64) 
    # 3. Numerical Stability: Clip mu to prevent log(0)
    mu = np.clip(mu, STABILITY_EPS, 1.0 - STABILITY_EPS)

    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iter):
        
        # --- E-Step ---
        
        # 1. Log Likelihood for each cluster: log P(x_i | z_i=k)
        log_mu = np.log(mu)
        log_one_minus_mu = np.log(1 - mu)
        
        # Matrix Multiplication: (N, D) @ (D, K) + (N, D) @ (D, K) -> (N, K)
        log_likelihoods = X @ log_mu.T + (1 - X) @ log_one_minus_mu.T
        
        # 2. Log Joint Probability: log P(x_i, z_i=k) = log(pi_k) + log P(x_i | z_i=k)
        log_pi = np.log(pi)
        log_joint_probs = log_likelihoods + log_pi 
        
        # 3. Log Evidence: log P(x_i) = log(sum_k exp(log P(x_i, z_i=k)))
        # Use Log-Sum-Exp to prevent Underflow
        log_evidence = log_sum_exp(log_joint_probs, axis=1)
        current_log_likelihood = np.sum(log_evidence)

        # 4. Responsibility (Gamma): gamma_ik = P(z_i=k | x_i)
        log_gamma = log_joint_probs - log_evidence
        gamma = np.exp(log_gamma)
        
        # --- M-Step ---
        
        # 1. Effective sample count N_k
        N_k = np.sum(gamma, axis=0)
        
        # 2. Update pi_k
        pi = N_k / num_samples
        
        # 3. Update mu_kj
        # weighted_sum_x = sum_i [ gamma_ik * x_ij ]
        weighted_sum_x = X.T @ gamma  # (num_features, K)
        mu = (weighted_sum_x / N_k).T # (K, num_features)
        
        # Numerical Stability: Clip mu again
        mu = np.clip(mu, STABILITY_EPS, 1.0 - STABILITY_EPS)
        
        # --- Convergence Check ---
        difference = current_log_likelihood - prev_log_likelihood
        print(f"No. of Iteration: {iteration + 1}, Log-Likelihood Diff: {difference:.10f}")
        
        if np.abs(difference) < tol:
            print(f"EM converged at iteration {iteration + 1}.")
            break
            
        prev_log_likelihood = current_log_likelihood
        
    # Final Cluster Assignment (Hard Assignment)
    y_pred_cluster = np.argmax(gamma, axis=1)
    
    return pi, mu, y_pred_cluster, gamma

# --- 5. EVALUATION AND OUTPUT FUNCTIONS ---

def assign_labels_and_evaluate(y_true, y_pred_cluster, mu, K, rows, cols):
    """
    Assigns labels using Majority Voting and prints the evaluation results.
    """
    
    # 1. Label Mapping (Majority Voting)
    cluster_to_label = {}
    
    for k in range(K):
        # Extract true labels for samples assigned to cluster k
        true_labels_in_cluster = y_true[y_pred_cluster == k]
        
        if len(true_labels_in_cluster) > 0:
            # Count occurrences of each true label
            counts = np.bincount(true_labels_in_cluster)
            # The most frequent true label is the assigned label (Majority Voting)
            assigned_label = np.argmax(counts)
            cluster_to_label[k] = assigned_label
        else:
            # Handle empty cluster
            cluster_to_label[k] = -1 
            
    # 2. Apply Mapping to get final predicted labels
    y_pred_final = np.array([cluster_to_label.get(k, -1) for k in y_pred_cluster])
    
    # 3. Output Results (for each digit)
    for true_digit in range(K):
        # Determine which Cluster ID was mapped to this true digit
        cluster_id = [k for k, v in cluster_to_label.items() if v == true_digit]
        
        # Binary Classification (Is True Digit vs. Isn't True Digit)
        y_true_binary = (y_true == true_digit).astype(int)
        y_pred_binary = (y_pred_final == true_digit).astype(int)
        
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        print(f"\nConfusion Matrix {true_digit}:")
        print(" " * 20 + f"Predict number {true_digit} Predict not number {true_digit}")
        print(f"Is number {true_digit}{TP:>20}{FN:>22}")
        print(f"Isn't number {true_digit}{FP:>20}{TN:>22}")
        print(f"Sensitivity (Successfully predict number {true_digit}): {sensitivity:.5f}")
        print(f"Specificity (Successfully predict not number {true_digit}): {specificity:.5f}")

        # Output Imagination of Numbers
        if cluster_id:
            # Use the mu from the cluster assigned to this digit
            best_mu = mu[cluster_id[0], :]
            print(f"\nLabeled class {true_digit} (from Cluster {cluster_id[0]}):")
            # Reshape the 784D vector back to 28x28 and format output (0.5 threshold)
            img = best_mu.reshape(rows, cols)
            for r in range(rows):
                print("".join(['1' if img[r, c] >= 0.5 else '0' for c in range(cols)]))
    
    # Total error rate
    total_error_rate = np.sum(y_true != y_pred_final) / len(y_true)
    print(f"\nTotal error rate: {total_error_rate:.10f}")


# --- 6. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # File names (assuming they match your screenshot and are in the current directory)
    IMAGE_FILE = 'train-images.idx3-ubyte_'
    LABEL_FILE = 'train-labels.idx1-ubyte_'
    K_CLUSTERS = 10 # Number of clusters
    MAX_ITER = 100 # Set a reasonable maximum number of iterations

    # 1. Load Data
    print("Loading and preprocessing data...")
    images, rows, cols = load_mnist_images(IMAGE_FILE)
    labels = load_mnist_labels(LABEL_FILE)
    
    # 2. Preprocessing: Binarization (using > 127 as the threshold)
    X_binary = preprocess_mnist_data(images)
    
    # 3. Execute EM Algorithm
    print(f"Starting EM Clustering with K={K_CLUSTERS}...")
    pi_final, mu_final, y_pred_cluster, gamma_final = bernoulli_em_clustering(
        X_binary, K_CLUSTERS, max_iter=MAX_ITER
    )
    
    # 4. Evaluate and Output Results
    print("\n--- Final Evaluation and Output ---")
    assign_labels_and_evaluate(labels, y_pred_cluster, mu_final, K_CLUSTERS, rows, cols)