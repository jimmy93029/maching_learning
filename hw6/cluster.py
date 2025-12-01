import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 0. 輔助函式：純 Numpy 計算距離矩陣
# ==========================================

def compute_sq_dist_mat(X, Y):
    """
    計算 X (N, d) 與 Y (M, d) 之間所有點對點的歐幾里得距離平方
    不使用 scipy.spatial.distance.cdist
    回傳: (N, M) 矩陣
    """
    # 1. 計算 X 每個點的長度平方 (sum of squares), 形狀轉為 (N, 1)
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    
    # 2. 計算 Y 每個點的長度平方, 形狀轉為 (1, M)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    
    # 3. 展開公式: ||x - y||^2 = x^2 + y^2 - 2*x*y
    # 利用廣播機制: (N, 1) + (1, M) -> (N, M)
    dist_mat = X_sq + Y_sq - 2 * np.dot(X, Y.T)
    
    # 4. 數值修正：因為浮點數誤差，有時候會出現極小的負數 (e.g. -1e-10)，修正為 0
    dist_mat[dist_mat < 0] = 0
    
    return dist_mat

# ==========================================
# 1. 基礎設定與讀取圖片
# ==========================================

def load_image(image_path):
    """讀取圖片並轉換為 data"""
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
    """將圖片合併為 GIF"""
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    if frames:
        frames[0].save(output_name, format="GIF", append_images=frames[1:],
                       save_all=True, duration=100, loop=0)
        print(f"GIF saved to {output_name}")

# ==========================================
# 2. Kernel 計算 (Spatial * Color)
# ==========================================

def compute_gram_matrix(colors, coords, gamma_c, gamma_s):
    """
    計算 Gram Matrix
    K(x, x') = exp(-gamma_s * ||S(x)-S(x')||^2) * exp(-gamma_c * ||C(x)-C(x')||^2)
    """
    print("Computing Spatial Kernel (Numpy)...")
    # 替換 cdist
    dist_s = compute_sq_dist_mat(coords, coords)
    K_s = np.exp(-gamma_s * dist_s)
    
    print("Computing Color Kernel (Numpy)...")
    # 替換 cdist
    dist_c = compute_sq_dist_mat(colors, colors)
    K_c = np.exp(-gamma_c * dist_c)
    
    K = K_s * K_c
    return K

# ==========================================
# 3. 一般 K-means (用於 Spectral Clustering)
# ==========================================

def initialize_centers(data, k, method='random'):
    """
    負責產生初始中心點，支援多種初始化方法
    data: (N, d) 資料矩陣
    k: 群數
    method: 'random', 'kmeans++', 'fps', 'random_partition'
    """
    n, dim = data.shape
    
    # 1. Random (Forgy Method): 隨機選 k 個點
    if method == 'random':
        indices = np.random.choice(n, k, replace=False)
        return data[indices]

    # 2. K-means++: 機率性選距離遠的點
    elif method == 'kmeans++':
        centers = []
        first_center_idx = np.random.randint(n)
        centers.append(data[first_center_idx])
        
        for _ in range(k - 1):
            current_centers = np.array(centers)
            # 計算到最近中心的距離
            dists = np.sum((data[:, None, :] - current_centers[None, :, :])**2, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # 機率分佈
            probs = min_dists / min_dists.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            ind = np.searchsorted(cumprobs, r)
            centers.append(data[ind])
        return np.array(centers)

    # 3. FPS (Farthest Point Sampling): 總是選最遠的點
    elif method == 'fps':
        centers = []
        # 第 1 個點隨機
        first_center_idx = np.random.randint(n)
        centers.append(data[first_center_idx])
        
        for _ in range(k - 1):
            current_centers = np.array(centers)
            # 計算距離
            dists = np.sum((data[:, None, :] - current_centers[None, :, :])**2, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # 直接選距離最大的那個點 (argmax)
            next_center_idx = np.argmax(min_dists)
            centers.append(data[next_center_idx])
            
        return np.array(centers)

    # 4. Random Partition: 隨機分群後算平均
    elif method == 'random_partition':
        # 隨機給每個點一個標籤 (0 ~ k-1)
        random_labels = np.random.randint(0, k, n)
        centers = np.zeros((k, dim))
        
        for i in range(k):
            mask = (random_labels == i)
            if np.any(mask):
                centers[i] = data[mask].mean(axis=0)
            else:
                # 萬一某一群沒分到點 (機率極低)，隨機選一個點補
                centers[i] = data[np.random.randint(n)]
        return centers

    else:
        raise ValueError(f"Unknown initialization method: {method}")


def simple_kmeans(U, k, h, w, output_dir='sc_frames', max_iters=100, init_method='random'):
    """
    對矩陣 U (N, d) 執行 K-means
    """
    n, dim = U.shape
    
    # 初始化
    centers = initialize_centers(U, k, init_method)
    prev_labels = np.zeros(n)
    
    for it in range(max_iters):

        # 計算距離矩陣 (N, k)
        # 這裡改用廣播 Broadcasting 來計算距離，不依賴 scipy
        # U: (N, 1, d), centers: (1, k, d) -> (N, k, d) -> sum sq -> (N, k)
        dists = np.sum((U[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        print(f"Kernel K-means Iteration {it+1}/{max_iters}")
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

# ==========================================
# 4. Kernel K-means 實作
# ==========================================

def kernel_k_means(K, k, h, w, max_iters=100, output_dir='kkm_frames'):
    """
    Kernel K-means 不需要算座標距離，只依賴 Kernel Matrix K
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    labels = np.random.randint(0, k, n)
    K_diag = np.diag(K)
    
    for it in range(max_iters):
        print(f"Kernel K-means Iteration {it+1}/{max_iters}")
        visualize_clusters(labels, h, w, f"{output_dir}/step_{it:03d}.png")
        
        dist_matrix = np.zeros((n, k))
        
        for c in range(k):
            mask = (labels == c)
            if not np.any(mask):
                continue
                
            N_c = np.sum(mask)
            
            # Kernel K-means 距離公式的三項
            # 1. K(x, x) -> K_diag
            # 2. -2 sum(K(x, xi)) / |C|
            term2 = -2 * np.sum(K[:, mask], axis=1) / N_c
            # 3. sum(sum(K(xi, xj))) / |C|^2
            term3 = np.sum(K[np.ix_(mask, mask)]) / (N_c ** 2)
            
            dist_matrix[:, c] = K_diag + term2 + term3
            
        new_labels = np.argmin(dist_matrix, axis=1)
        
        if np.all(new_labels == labels):
            print("Converged!")
            break
        labels = new_labels
        
    return labels

# ==========================================
# 5. Spectral Clustering 實作
# ==========================================

def spectral_clustering(K, k, h, w, mode='ratio', output_dir='sc_frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n = K.shape[0]
    W = K
    D_vec = np.sum(W, axis=1)
    D_vec[D_vec == 0] = 1e-10 
    D = np.diag(D_vec)
    L = D - W
    
    print(f"Computing Eigenvectors for {mode} cut...")
    
    # 這裡只用到 numpy.linalg，這是允許的
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
        # 避免除以 0
        row_sums[row_sums == 0] = 1e-10
        U = U / row_sums
        
    print("Running K-means on Eigenvectors...")
    labels = simple_kmeans(U, k, h, w, output_dir)
    
    return labels

# ==========================================
# 6. 視覺化工具
# ==========================================

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
# 主程式
# ==========================================

if __name__ == "__main__":
    IMAGE_FILE = 'image1.png' # 請確認檔名
    GAMMA_C = 1
    GAMMA_S = 1
    K_CLUSTERS = 2
    
    if os.path.exists(IMAGE_FILE):
        data, colors, coords, h, w = load_image(IMAGE_FILE)
        
        # 1. 計算 Kernel (純 Numpy)
        Gram_K = compute_gram_matrix(colors, coords, GAMMA_C, GAMMA_S)
        
        # 2. Kernel K-means
        # print("\n--- Kernel K-means ---")
        # kernel_k_means(Gram_K, K_CLUSTERS, h, w, output_dir=f'results_npy/kkm')
        # make_gif(f'results_npy/kkm', 'kkm_numpy.gif')
        
        # 3. Spectral (Ratio)
        print("\n--- Ratio Cut ---")
        spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='ratio', output_dir=f'results_npy/ratio')
        make_gif(f'results_npy/ratio', 'ratio_numpy.gif')
        
        # 4. Spectral (Normalized)
        print("\n--- Normalized Cut ---")
        spectral_clustering(Gram_K, K_CLUSTERS, h, w, mode='normalized', output_dir=f'results_npy/norm')
        make_gif(f'results_npy/norm', 'norm_numpy.gif')
        
    else:
        print(f"Error: {IMAGE_FILE} not found.")