import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import shutil

# ==========================================
# 0. 輔助函式：純 Numpy 計算距離矩陣
# ==========================================

def compute_sq_dist_mat(X, Y):
    """
    計算 X (N, d) 與 Y (M, d) 之間所有點對點的歐幾里得距離平方
    """
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    dist_mat = X_sq + Y_sq - 2 * np.dot(X, Y.T)
    dist_mat[dist_mat < 0] = 0
    return dist_mat

# ==========================================
# 1. 初始化函式 (核心修改)
# ==========================================

def initialize_centers(data, k, method='random', for_kernel=False, K_matrix=None):
    """
    負責產生初始中心點 (Standard) 或 初始標籤 (Kernel)
    
    Args:
        data: (N, d) 資料矩陣 (Standard mode)
        k: 群數
        method: 'random', 'kmeans++', 'fps', 'random_partition'
        for_kernel: Boolean, 是否為 Kernel K-means 模式
        K_matrix: (N, N) Gram Matrix, 當 for_kernel=True 時必須提供
        
    Returns:
        if for_kernel=False: centers (k, d) - 初始中心座標
        if for_kernel=True:  labels (N,)    - 初始分群標籤
    """
    n = data.shape[0] if data is not None else K_matrix.shape[0]
    
    # --- Logic for Kernel K-means (Return Labels) ---
    if for_kernel:
        if K_matrix is None:
            raise ValueError("K_matrix must be provided when for_kernel=True")
            
        K_diag = np.diag(K_matrix) 
        
        # 1. Random Partition: 直接隨機給標籤
        if method == 'random_partition':
            return np.random.randint(0, k, n)
            
        # For 'random', 'kmeans++', 'fps': We first pick k "seed indices"
        centers_indices = []
        
        # Step A: Pick Seed Points
        if method == 'random':
            centers_indices = np.random.choice(n, k, replace=False)
            
        elif method in ['kmeans++', 'fps']:
            first_idx = np.random.randint(n)
            centers_indices.append(first_idx)
            
            for _ in range(k - 1):
                min_dists = np.full(n, np.inf)
                for c_idx in centers_indices:
                    # Kernel Distance: d^2(x, c) = K(x,x) + K(c,c) - 2K(x,c)
                    dists_to_c = K_diag + K_diag[c_idx] - 2 * K_matrix[:, c_idx]
                    min_dists = np.minimum(min_dists, dists_to_c)
                
                min_dists = np.maximum(min_dists, 0) # Fix numerical error

                if method == 'kmeans++':
                    if min_dists.sum() == 0:
                        probs = np.ones(n) / n
                    else:
                        probs = min_dists / min_dists.sum()
                    cumprobs = np.cumsum(probs)
                    r = np.random.rand()
                    next_idx = np.searchsorted(cumprobs, r)
                    centers_indices.append(next_idx)
                    
                elif method == 'fps':
                    next_idx = np.argmax(min_dists)
                    centers_indices.append(next_idx)

        # Step B: Convert Seeds to Labels
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
                dists = np.sum((data[:, None, :] - current_centers[None, :, :])**2, axis=2)
                min_dists = np.min(dists, axis=1)
                
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
                next_center_idx = np.argmax(min_dists)
                centers.append(data[next_center_idx])
            return np.array(centers)

        elif method == 'random_partition':
            random_labels = np.random.randint(0, k, n)
            centers = np.zeros((k, dim))
            for i in range(k):
                mask = (random_labels == i)
                if np.any(mask):
                    centers[i] = data[mask].mean(axis=0)
                else:
                    centers[i] = data[np.random.randint(n)]
            return centers
            
        else:
            raise ValueError(f"Unknown method: {method}")

# ==========================================
# 2. 基礎設定與讀取圖片
# ==========================================

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100)) 
    data = np.array(img) / 255.0
    h, w, c = data.shape
    
    # 建立座標網格 (Spatial)
    coords = np.zeros((h * w, 2))
    for i in range(h):
        for j in range(w):
            coords[i * w + j] = [i / h, j / w]
            
    # 顏色資料 (Color)
    colors = data.reshape(-1, 3)
    return data, colors, coords, h, w

def make_gif(frame_folder, output_name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    if frames:
        frames[0].save(output_name, format="GIF", append_images=frames[1:],
                       save_all=True, duration=100, loop=0)
        print(f"GIF saved to {output_name}")

def visualize_clusters(labels, h, w, filename):
    cluster_colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ])
    img_data = np.zeros((h * w, 3))
    for i in range(h * w):
        img_data[i] = cluster_colors[labels[i] % len(cluster_colors)]
    img_data = img_data.reshape(h, w, 3)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img_data)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==========================================
# 3. Kernel 計算
# ==========================================

def compute_gram_matrix(colors, coords, gamma_c, gamma_s):
    print("Computing Spatial Kernel (Numpy)...")
    dist_s = compute_sq_dist_mat(coords, coords)
    K_s = np.exp(-gamma_s * dist_s)
    
    print("Computing Color Kernel (Numpy)...")
    dist_c = compute_sq_dist_mat(colors, colors)
    K_c = np.exp(-gamma_c * dist_c)
    
    K = K_s * K_c
    return K

# ==========================================
# 4. Clustering Algorithms
# ==========================================

def simple_kmeans(U, k, h, w, output_dir, max_iters=100, init_method='random'):
    """
    Standard K-means applied on Eigenvectors U
    """
    n, dim = U.shape
    
    # --- Modified: Use initialize_centers ---
    centers = initialize_centers(U, k, method=init_method, for_kernel=False)
    
    prev_labels = np.zeros(n)
    
    for it in range(max_iters):
        dists = np.sum((U[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Print less frequently to avoid clutter
        if it % 10 == 0: print(f"  Standard K-means iter {it}")
        visualize_clusters(labels, h, w, f"{output_dir}/step_{it:03d}.png")
        
        if np.all(labels == prev_labels):
            break
        prev_labels = labels.copy()
        
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                centers[i] = U[mask].mean(axis=0)
            else:
                centers[i] = U[np.random.randint(n)]
                
    return labels


def kernel_k_means(K, k, h, w, max_iters=100, output_dir='kkm_frames', init_method='random'):
    """
    Kernel K-means with initialization support
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    
    # --- Modified: Use initialize_centers for Kernel ---
    print(f"Initializing Kernel K-means with {init_method}...")
    labels = initialize_centers(data=None, k=k, method=init_method, for_kernel=True, K_matrix=K)
    
    K_diag = np.diag(K)
    
    for it in range(max_iters):
        print(f"  Kernel K-means Iteration {it+1}/{max_iters}")
        visualize_clusters(labels, h, w, f"{output_dir}/step_{it:03d}.png")
        
        dist_matrix = np.zeros((n, k))
        
        for c in range(k):
            mask = (labels == c)
            if not np.any(mask):
                continue
            
            N_c = np.sum(mask)
            term2 = -2 * np.sum(K[:, mask], axis=1) / N_c
            term3 = np.sum(K[np.ix_(mask, mask)]) / (N_c ** 2)
            dist_matrix[:, c] = K_diag + term2 + term3
            
        new_labels = np.argmin(dist_matrix, axis=1)
        
        if np.all(new_labels == labels):
            print("  Converged!")
            visualize_clusters(labels, h, w, f"{output_dir}/converged.png")
            break
        labels = new_labels
        
    return labels


def spectral_clustering(K, k, h, w, mode, output_dir, init_method='random'):
    """
    Spectral Clustering Wrapper
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    W = K
    D_vec = np.sum(W, axis=1)
    D_vec[D_vec == 0] = 1e-10 
    D = np.diag(D_vec)
    L = D - W
    
    print(f"Computing Eigenvectors for {mode} cut...")
    
    if mode == 'ratio':
        eigvals, eigvecs = np.linalg.eigh(L)
        U = eigvecs[:, 1:k+1]
        
    elif mode == 'normalized':
        D_sqrt_inv = np.diag(1.0 / np.sqrt(D_vec))
        L_sym = np.dot(np.dot(D_sqrt_inv, L), D_sqrt_inv)
        
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        U = eigvecs[:, :k]
        
        # Row normalization
        row_sums = np.sqrt(np.sum(U**2, axis=1)).reshape(-1, 1)
        row_sums[row_sums == 0] = 1e-10
        U = U / row_sums
        
    print(f"Running K-means on Eigenvectors (Init: {init_method})...")
    # Pass init_method to simple_kmeans
    labels = simple_kmeans(U, k, h, w, output_dir, init_method=init_method)
    
    return labels

# ==========================================
# Main: Run 4x3 Experiments
# ==========================================

if __name__ == "__main__":
    IMAGE_FILE = 'image1.png'  # Ensure this file exists
    GAMMA_C = 1          
    GAMMA_S = 1
    K_CLUSTERS = 3
    
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: {IMAGE_FILE} not found. Please place an image file.")
    else:
        # Load and Precompute Gram Matrix once
        data, colors, coords, h, w = load_image(IMAGE_FILE)
        Gram_K = compute_gram_matrix(colors, coords, GAMMA_C, GAMMA_S)
        
        # Define experiments
        init_methods = ['random', 'kmeans++', 'fps', 'random_partition']
        base_output_dir = f"experiments/{IMAGE_FILE}/K_CLUSTERS={K_CLUSTERS}/GAMMA_C={GAMMA_C}_GAMMA_S={GAMMA_S}"
        
        if os.path.exists(base_output_dir):
            shutil.rmtree(base_output_dir) # Clean previous results
            
        print(f"\nStarting 4 x 3 Experiments...")
        print("="*40)

        for method in init_methods:
            print(f"\n>>> Testing Initialization: {method.upper()}")
            
            # 1. Kernel K-means
            print(f"  [1/3] Kernel K-means ({method})")
            dir_kkm = f"{base_output_dir}/{method}/kkm"
            kernel_k_means(Gram_K, K_CLUSTERS, h, w, output_dir=dir_kkm, init_method=method)
            make_gif(dir_kkm, f"{base_output_dir}/{method}_kkm.gif")
            
            # 2. Spectral (Ratio)
            print(f"  [2/3] Spectral Ratio Cut ({method})")
            dir_ratio = f"{base_output_dir}/{method}/ratio"
            spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='ratio', output_dir=dir_ratio, init_method=method)
            make_gif(dir_ratio, f"{base_output_dir}/{method}_ratio.gif")
            
            # 3. Spectral (Normalized)
            print(f"  [3/3] Spectral Normalized Cut ({method})")
            dir_norm = f"{base_output_dir}/{method}/norm"
            spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='normalized', output_dir=dir_norm, init_method=method)
            make_gif(dir_norm, f"{base_output_dir}/{method}_norm.gif")
            
        print("\nAll experiments completed.")