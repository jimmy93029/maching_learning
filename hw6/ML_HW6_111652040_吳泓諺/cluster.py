import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import shutil
import time

# ==========================================
# 0. Helper Function: Pure Numpy Distance Matrix
# ==========================================

def compute_sq_dist_mat(X, Y):
    """
    Computes the squared Euclidean distance matrix between X (N, d) and Y (M, d).
    Uses the matrix identity: ||A - B||^2 = ||A||^2 + ||B||^2 - 2AB^T
    """
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    dist_mat = X_sq + Y_sq - 2 * np.dot(X, Y.T)
    # Numerical stability fix
    dist_mat[dist_mat < 0] = 0
    return dist_mat

# ==========================================
# 1. Initialization Function (Core Modification)
# ==========================================

def initialize_centers(data, k, method='random', for_kernel=False, K_matrix=None):
    n = data.shape[0] if data is not None else K_matrix.shape[0]
    
    # --- Logic for Kernel K-means (Return Labels) ---
    if for_kernel:
        if K_matrix is None:
            raise ValueError("K_matrix must be provided when for_kernel=True")
            
        K_diag = np.diag(K_matrix) 
        
        # 1. Random Partition: Randomly assign labels
        if method == 'random_partition':
            return np.random.randint(0, k, n)
            
        # 2. Select k seed indices
        centers_indices = []
        
        if method == 'random':
            centers_indices = np.random.choice(n, k, replace=False)
            
        elif method in ['kmeans++', 'fps']:
            first_idx = np.random.randint(n)
            centers_indices.append(first_idx)
            
            for _ in range(k - 1):
                # Kernel Distance Squared: d^2(x, c) = K(x,x) + K(c,c) - 2K(x,c)
                min_dists = np.full(n, np.inf)
                for c_idx in centers_indices:
                    dists_to_c = K_diag + K_diag[c_idx] - 2 * K_matrix[:, c_idx]
                    min_dists = np.minimum(min_dists, dists_to_c)
                
                min_dists = np.maximum(min_dists, 0) # Fix numerical error

                if method == 'kmeans++':
                    # Weighted probabilistic selection based on distance squared
                    if min_dists.sum() == 0:
                        probs = np.ones(n) / n
                    else:
                        probs = min_dists / min_dists.sum()
                    cumprobs = np.cumsum(probs)
                    r = np.random.rand()
                    next_idx = np.searchsorted(cumprobs, r)
                    centers_indices.append(next_idx)
                    
                elif method == 'fps':
                    # Select the point farthest from all current centers
                    next_idx = np.argmax(min_dists)
                    centers_indices.append(next_idx)

        # 3. Assign all points to the closest seed point (convert to initial labels)
        dist_matrix = np.zeros((n, k))
        for i, c_idx in enumerate(centers_indices):
            dist_matrix[:, i] = K_diag + K_diag[c_idx] - 2 * K_matrix[:, c_idx]
            
        initial_labels = np.argmin(dist_matrix, axis=1)
        return initial_labels

    # --- Logic for Standard K-means (Return Coordinates) ---
    else:
        n, dim = data.shape
        
        if method == 'random':
            indices = np.random.choice(n, k, replace=False)
            return data[indices]

        elif method == 'kmeans++':
            centers = []
            first_center_idx = np.random.randint(n)
            centers.append(data[first_center_idx])
            
            for _ in range(k - 1):
                current_centers = np.array(centers)
                # Calculate squared distance from all points to all current centers
                dists = np.sum((data[:, None, :] - current_centers[None, :, :])**2, axis=2)
                min_dists = np.min(dists, axis=1) # Squared distance to the closest center
                
                # Weighted probabilistic selection based on distance squared
                if min_dists.sum() == 0:
                     probs = np.ones(n) / n
                else:
                    probs = min_dists / min_dists.sum()
                cumprobs = np.cumsum(probs)
                r = np.random.rand()
                ind = np.searchsorted(cumprobs, r)
                centers.append(data[ind])
            return np.array(centers)

        elif method == 'fps':
            centers = []
            first_center_idx = np.random.randint(n)
            centers.append(data[first_center_idx])
            
            for _ in range(k - 1):
                current_centers = np.array(centers)
                dists = np.sum((data[:, None, :] - current_centers[None, :, :])**2, axis=2)
                min_dists = np.min(dists, axis=1)
                # Select the farthest point
                next_center_idx = np.argmax(min_dists)
                centers.append(data[next_center_idx])
            return np.array(centers)

        elif method == 'random_partition':
            # Randomly partition and compute the mean of each partition as the center
            random_labels = np.random.randint(0, k, n)
            centers = np.zeros((k, dim))
            for i in range(k):
                mask = (random_labels == i)
                if np.any(mask):
                    centers[i] = data[mask].mean(axis=0)
                else:
                    # Handle empty cluster
                    centers[i] = data[np.random.randint(n)]
            return centers
            
        else:
            raise ValueError(f"Unknown method: {method}")

# ==========================================
# 2. Utils
# ==========================================

def load_image(image_path):
    """Loads the image and prepares Color and Spatial Coordinates data."""
    img = Image.open(image_path)
    img = img.resize((100, 100)) # Resize to 100x100
    data = np.array(img) / 255.0
    h, w, c = data.shape
    
    # Create normalized spatial coordinates grid (Spatial)
    coords = np.zeros((h * w, 2))
    for i in range(h):
        for j in range(w):
            coords[i * w + j] = [i / h, j / w]
            
    # Flattened color data (Color)
    colors = data.reshape(-1, 3)
    return data, colors, coords, h, w

# ==========================================
# 3. Kernel Computation
# ==========================================

def compute_gram_matrix(colors, coords, gamma_c, gamma_s):
    """
    Computes the composite Gram matrix K = K_s * K_c (Hadamard product).
    Uses the Gaussian Kernel: k(x, y) = exp(-gamma * d(x, y)^2)
    """
    print("Computing Spatial Kernel (K_s)...")
    dist_s = compute_sq_dist_mat(coords, coords)
    K_s = np.exp(-gamma_s * dist_s)
    
    print("Computing Color Kernel (K_c)...")
    dist_c = compute_sq_dist_mat(colors, colors)
    K_c = np.exp(-gamma_c * dist_c)
    
    K = K_s * K_c # Composite Kernel
    return K

# ==========================================
# 4. Clustering Algorithms
# ==========================================

def simple_kmeans(U, k, h, w, output_dir, max_iters=100, init_method='random'):
    """
    Standard K-means applied on the Eigenvectors U (used in the final step of Spectral Clustering).
    """
    n, dim = U.shape
    
    # Initialize centers
    centers = initialize_centers(U, k, method=init_method, for_kernel=False)
    
    prev_labels = np.zeros(n)
    
    for it in range(max_iters):
        # E-step: Assign labels (minimize Euclidean distance)
        dists = np.sum((U[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Print and visualize less frequently to reduce overhead
        if it % 10 == 0: print(f"  Standard K-means iter {it}")
        visualize_clusters(labels, h, w, f"{output_dir}/step_{it:03d}.png")
        
        if np.all(labels == prev_labels):
            break
        prev_labels = labels.copy()
        
        # M-step: Update centers (compute mean)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                centers[i] = U[mask].mean(axis=0)
            else:
                # Handle empty cluster
                centers[i] = U[np.random.randint(n)]
                
    return labels


def kernel_k_means(K, k, h, w, max_iters=100, output_dir='kkm_frames', init_method='random'):
    """
    Kernel K-means (KKM). Clusters data in the implicit high-dimensional feature space.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    
    # Initialize labels using the specified method (for_kernel=True)
    print(f"Initializing Kernel K-means with {init_method}...")
    labels = initialize_centers(data=None, k=k, method=init_method, for_kernel=True, K_matrix=K)
    
    K_diag = np.diag(K)
    
    for it in range(max_iters):
        print(f"  Kernel K-means Iteration {it+1}/{max_iters}")
        visualize_clusters(labels, h, w, f"{output_dir}/step_{it:03d}.png")
        
        dist_matrix = np.zeros((n, k))
        
        # E-step: Assign labels (minimize kernel distance)
        for c in range(k):
            mask = (labels == c)
            if not np.any(mask):
                continue
            
            N_c = np.sum(mask)
            # Calculate squared distance using the kernel trick: d^2(x, mu_c) = K(x,x) + term2 + term3
            term2 = -2 * np.sum(K[:, mask], axis=1) / N_c
            term3 = np.sum(K[np.ix_(mask, mask)]) / (N_c ** 2)
            dist_matrix[:, c] = K_diag + term2 + term3
            
        new_labels = np.argmin(dist_matrix, axis=1)
        
        if np.all(new_labels == labels):
            print("  Converged!")
            visualize_clusters(labels, h, w, f"{output_dir}/converged.png")
            break
        labels = new_labels
        
    return labels


def spectral_clustering(K, k, h, w, mode, base_output_dir, output_dir, init_method='random'):
    """
    Spectral Clustering. Performs dimensionality reduction using Graph Laplacian eigenvectors 
    and then clusters the projected data using K-means.
    Args:
        mode: 'ratio' (Ratio Cut) or 'normalized' (Normalized Cut)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    W = K # W is the similarity/weight matrix
    
    # 1. Construct Graph Laplacian (L)
    D_vec = np.sum(W, axis=1)
    D_vec[D_vec == 0] = 1e-10 
    D = np.diag(D_vec) # Degree Matrix
    L = D - W           # Unnormalized Laplacian Matrix
    
    # 2. Eigendecomposition based on mode
    if mode == 'ratio':
        # Ratio Cut: L*u = lambda*u
        eigvals, eigvecs = np.linalg.eigh(L)
        # Select k smallest non-zero eigenvectors (usually starting from the second one)
        U = eigvecs[:, 1:k+1]
        
    elif mode == 'normalized':
        # Normalized Cut: L_sym*u = lambda*u
        D_sqrt_inv = np.diag(1.0 / np.sqrt(D_vec))
        L_sym = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv) # Symmetric Normalized Laplacian
        
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Select k smallest eigenvectors
        U = eigvecs[:, :k]
        
        # Row normalization
        row_sums = np.sqrt(np.sum(U**2, axis=1)).reshape(-1, 1)
        row_sums[row_sums == 0] = 1e-10
        U = U / row_sums

    # 3. Perform K-means on the feature space U
    labels = simple_kmeans(U, k, h, w, output_dir, init_method=init_method)
    plot_eigenspace(U, labels, mode, base_output_dir, k)
    
    return labels

# ==========================================
# Plot
# ==========================================

def plot_eigenspace(U, labels, mode, output_dir, k):
    """
    Plots the data points in the Eigenspace (first two dimensions) 
    to visualize the cluster separation.
    """
    if k < 2:
        print("K is too small to plot 2D eigenspace.")
        return
    
    X_coord = U[:, 0]
    Y_coord = U[:, 1]
    
    # Define color palette
    cluster_colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ])
    
    # Map labels to colors
    colors = cluster_colors[labels % len(cluster_colors)]
    
    # --- Plotting ---
    plt.figure(figsize=(8, 8))
    plt.scatter(X_coord, Y_coord, c=colors, s=5, alpha=0.5)
    
    plt.title(f"Eigenspace Scatter Plot ({mode} Cut, k={k})")
    plt.xlabel(f"Eigenvector 1 (U[:, 0])")
    plt.ylabel(f"Eigenvector 2 (U[:, 1])")
    
    # Save the figure
    filename = f"{output_dir}/eigenspace_plot_{mode}_cluster{k}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Eigenspace plot saved to {filename}")


def make_gif(frame_folder, output_name):
    """Creates a GIF from image frames showing the clustering process."""
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    if frames:
        frames[0].save(output_name, format="GIF", append_images=frames[1:],
                       save_all=True, duration=100, loop=0)
        print(f"GIF saved to {output_name}")

def visualize_clusters(labels, h, w, filename):
    """Converts cluster labels into a visual image and saves it."""
    cluster_colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ])
    img_data = np.zeros((h * w, 3))
    # Map label i to the corresponding color
    for i in range(h * w):
        img_data[i] = cluster_colors[labels[i] % len(cluster_colors)]
    img_data = img_data.reshape(h, w, 3)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img_data)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# ==========================================
# Main: Run 4x3 Experiments
# ==========================================

if __name__ == "__main__":
    IMAGE_FILE = 'image1.png' # Ensure this file exists
    GAMMA_C = 1               # Color Kernel parameter
    GAMMA_S = 1               # Spatial Kernel parameter
    K_CLUSTERS = 3            # Number of clusters K
    
    timing_results = {}

    if not os.path.exists(IMAGE_FILE):
        print(f"Error: {IMAGE_FILE} not found. Please place an image file.")
    else:
        # --- Precomputation ---
        start_precomp = time.time()
        data, colors, coords, h, w = load_image(IMAGE_FILE)
        Gram_K = compute_gram_matrix(colors, coords, GAMMA_C, GAMMA_S)
        end_precomp = time.time()
        precomp_time = end_precomp - start_precomp
        print(f"✅ Precomputation time (Load + Gram Matrix): **{precomp_time:.4f} seconds**")
        
        # Define experiment parameters
        init_methods = ['random', 'kmeans++', 'fps', 'random_partition']
        base_output_dir = f"experiments/{IMAGE_FILE}/K_CLUSTERS={K_CLUSTERS}/GAMMA_C={GAMMA_C}_GAMMA_S={GAMMA_S}"
        
        if os.path.exists(base_output_dir):
            shutil.rmtree(base_output_dir) # Clean previous results
            
        print(f"\nStarting 4 x 3 Experiments...")
        print("="*40)
        
        start_all_experiments = time.time() 

        for method in init_methods:
            timing_results[method] = {}
            print(f"\n>>> Testing Initialization: {method.upper()}")
            
            # 1. Kernel K-means (KKM)
            print(f"  [1/3] Kernel K-means ({method})")
            dir_kkm = f"{base_output_dir}/{method}/kkm"
            start_kkm = time.time()
            kernel_k_means(Gram_K, K_CLUSTERS, h, w, output_dir=dir_kkm, init_method=method)
            make_gif(dir_kkm, f"{base_output_dir}/{method}_kkm.gif")
            end_kkm = time.time()
            timing_results[method]['kkm'] = end_kkm - start_kkm
            print(f"    -> Execution Time: **{timing_results[method]['kkm']:.4f} seconds**")
            
            # 2. Spectral (Ratio Cut)
            print(f"  [2/3] Spectral Ratio Cut ({method})")
            dir_ratio = f"{base_output_dir}/{method}/ratio"
            start_ratio = time.time()
            spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='ratio', base_output_dir=base_output_dir, output_dir=dir_ratio, init_method=method)
            make_gif(dir_ratio, f"{base_output_dir}/{method}_ratio.gif")
            end_ratio = time.time()
            timing_results[method]['ratio'] = end_ratio - start_ratio
            print(f"    -> Execution Time: **{timing_results[method]['ratio']:.4f} seconds**")

            # 3. Spectral (Normalized Cut)
            print(f"  [3/3] Spectral Normalized Cut ({method})")
            dir_norm = f"{base_output_dir}/{method}/norm"
            start_norm = time.time()
            spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='normalized', base_output_dir=base_output_dir, output_dir=dir_norm, init_method=method)
            make_gif(dir_norm, f"{base_output_dir}/{method}_norm.gif")
            end_norm = time.time()
            timing_results[method]['norm'] = end_norm - start_norm
            print(f"    -> Execution Time: **{timing_results[method]['norm']:.4f} seconds**")
            
        # --- Total Experiment Timing ---
        end_all_experiments = time.time()
        total_experiment_time = end_all_experiments - start_all_experiments
        print("\n" + "="*40)
        print(f"Total time for all 12 experiments: **{total_experiment_time:.4f} seconds**")
        print("\nAll experiments completed.")
        
        # --- Result Summary Table ---
        print("\n📊 **Experiment Time Summary (seconds)**")
        print("| Initialization Method | Kernel K-means | Spectral Ratio | Spectral Normalized |")
        print("|:----------------------|:--------------:|:--------------:|:--------------------|")
        for method in init_methods:
            kkm_time = timing_results[method].get('kkm', 0)
            ratio_time = timing_results[method].get('ratio', 0)
            norm_time = timing_results[method].get('norm', 0)
            print(f"| **{method.title()}** | {kkm_time:.4f} | {ratio_time:.4f} | {norm_time:.4f} |")
        print(f"| **Precomputation** | **{precomp_time:.4f}** | | |")