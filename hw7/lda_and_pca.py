import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2
import os


def load_yale_data(folder_path):
    images = []
    labels = []
    
    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        if filename.endswith(".pgm"):
            # 讀取影像並轉為灰階
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 將影像拉平成一維向量
                images.append(img.flatten())
                # 假設檔名前半部是類別編號 (如 subject01.sad -> 1)
                subject_id = int(filename.replace("subject", "").split(".")[0])
                labels.append(subject_id)
                
    return np.array(images, dtype=np.float64), np.array(labels)


def perform_pca(X, n_components):
    # 1. 中心化 (n_samples, n_features)
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    
    # 2. 轉置技巧：計算 X @ X.T (n_samples, n_samples)
    # 這裡矩陣大小只有 135x135 左右，運算極快且省記憶體
    # 注意：這裡不要用 np.cov，直接用矩陣相乘
    gram_matrix = np.dot(X_centered, X_centered.T)
    
    # 3. 計算特徵分解
    eigenvalues, eigenvectors_small = np.linalg.eigh(gram_matrix)
    
    # 4. 排序特徵值 (從大到小)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_small = eigenvectors_small[:, idx]
    
    # 5. 映射回原始高維空間：W = X_centered.T @ V_small
    # 這是為了得到長度為 45045 的特徵向量
    W_pca = np.dot(X_centered.T, eigenvectors_small[:, :n_components])
    
    # 6. 重要：單位化 (Normalization)
    # 因為點積出來的向量長度不為 1，必須手動單位化
    for i in range(W_pca.shape[1]):
        W_pca[:, i] = W_pca[:, i] / np.linalg.norm(W_pca[:, i])
        
    # 投影數據到 PCA 空間
    X_pca = np.dot(X_centered, W_pca)
    
    return X_pca, W_pca, mean_face


import numpy as np

def perform_lda(X, y, n_components):
    """
    純 LDA 轉置技巧版：避免分配 (d, d) 矩陣空間
    X: (n_samples, n_features) -> 例如 (135, 45045)
    y: 標籤
    """
    n_samples, n_features = X.shape
    class_labels = np.unique(y)
    
    # 1. 中心化與基礎計算
    mean_overall = np.mean(X, axis=0)
    X_centered = X - mean_overall  # (N, d)
    
    # 我們假設 w = X_centered.T @ alpha
    # 目標是求解 (X_centered @ SB @ X_centered.T) alpha = lambda (X_centered @ SW @ X_centered.T) alpha
    # 令 Kb = X_centered @ SB @ X_centered.T (N, N)
    # 令 Kw = X_centered @ SW @ X_centered.T (N, N)
    
    Kw = np.zeros((n_samples, n_samples))
    Kb = np.zeros((n_samples, n_samples))
    
    for c in class_labels:
        # 取出該類別的樣本索引
        idx_c = (y == c)
        X_c = X[idx_c]
        n_c = X_c.shape[0]
        
        # 該類別中心相對於全局中心的投影向量 (N, 1)
        mean_c = np.mean(X_c, axis=0)
        # v_b 是類別中心在樣本空間中的代表
        v_b = np.dot(X_centered, (mean_c - mean_overall).T).reshape(-1, 1)
        Kb += n_c * np.dot(v_b, v_b.T)
        
        # 類內樣本相對於類別中心的投影
        # v_w 矩陣大小為 (N, n_c)
        v_w = np.dot(X_centered, (X_c - mean_c).T)
        Kw += np.dot(v_w, v_w.T)

    # 2. 求解 N x N 的廣義特徵值問題
    # 為了穩定性，對 Kw 加上微小值
    Kw += np.eye(n_samples) * 1e-6
    
    # 求解 Kw^-1 @ Kb
    A = np.dot(np.linalg.pinv(Kw), Kb)
    eigenvalues, alphas = np.linalg.eig(A)
    
    # 3. 排序並選取 alpha
    idx = np.argsort(eigenvalues.real)[::-1]
    alphas = alphas.real[:, idx][:, :n_components]
    
    # 4. 映射回原始空間：W = X_centered.T @ alpha
    # W 的維度會是 (d, n_components)，即 (45045, n_components)
    W_lda = np.dot(X_centered.T, alphas)
    
    # 5. 單位化特徵向量
    for i in range(W_lda.shape[1]):
        W_lda[:, i] /= np.linalg.norm(W_lda[:, i])
    
    # 投影數據
    X_lda = np.dot(X_centered, W_lda)
    
    return X_lda, W_lda


def testing(train_X, train_y, test_X, test_y, W, mean_face, method_name, img_shape=(231, 195)):
    """
    純 NumPy 實作：同時執行辨識測試 (Recognition) 與 重建測試 (Reconstruction)
    """
    # 建立目錄
    recog_dir = f'experiments/{method_name}/recognition/'
    recon_dir = f'experiments/{method_name}/reconstruction/'
    for d in [recog_dir, recon_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # --- 共同步驟：投影 ---
    train_X_centered = train_X - mean_face
    test_X_centered = test_X - mean_face
    
    train_features = np.dot(train_X_centered, W)  
    test_features = np.dot(test_X_centered, W)    

    # ==========================================
    # A. 辨識測試 (使用 NumPy 替換 cdist)
    # ==========================================
    # 1. 計算各自的平方和 (L2 Norm Squared)
    # test_sq: (M, 1), train_sq: (1, N)
    test_sq = np.sum(test_features**2, axis=1)[:, np.newaxis]
    train_sq = np.sum(train_features**2, axis=1)[np.newaxis, :]
    
    # 2. 計算點積 (Dot Product): (M, N)
    dot_product = np.dot(test_features, train_features.T)
    
    # 3. 組合公式: dists^2 = a^2 + b^2 - 2ab
    # 使用 np.maximum 確保數值穩定，避免出現極小的負數導致開根號出錯
    dists_sq = np.maximum(test_sq + train_sq - 2 * dot_product, 0)
    dists = np.sqrt(dists_sq)
    
    closest_idx = np.argmin(dists, axis=1)
    predictions = train_y[closest_idx]
    
    accuracy = np.mean(predictions == test_y)
    print(f"[{method_name.upper()}] Recognition Accuracy: {accuracy * 100:.2f}%")

    # ==========================================
    # B. 重建測試 (Reconstruction)
    # ==========================================
    test_X_reconstructed = np.dot(test_features, W.T) + mean_face
    mse = np.mean((test_X - test_X_reconstructed) ** 2)
    print(f"[{method_name.upper()}] Reconstruction MSE: {mse:.4f}")

    # --- 視覺化輸出 ---
    for i in range(min(5, len(test_X))):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(test_X[i].reshape(img_shape), cmap='gray')
        plt.title(f"Original (ID:{test_y[i]})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(train_X[closest_idx[i]].reshape(img_shape), cmap='gray')
        plt.title(f"Predicted Match (ID:{predictions[i]})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(test_X_reconstructed[i].reshape(img_shape), cmap='gray')
        plt.title(f"Reconstructed (MSE:{np.mean((test_X[i]-test_X_reconstructed[i])**2):.2f})")
        plt.axis('off')

        plt.savefig(os.path.join(recon_dir, f"compare_{i}.png"))
        plt.close()



def compute_kernel(X1, X2, kernel_type='linear', gamma=None):
    """
    計算核矩陣 K(X1, X2)
    X1: (M, d), X2: (N, d) -> K: (M, N)
    """
    if kernel_type == 'linear':
        return np.dot(X1, X2.T)
    
    elif kernel_type == 'rbf':
        if gamma is None:
            gamma = 1.0 / X1.shape[1]
        # 使用 dist^2 = a^2 + b^2 - 2ab 避開 scipy.cdist
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = sq1 + sq2 - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * np.maximum(dist_sq, 0))
    
    else:
        raise ValueError("Unsupported kernel type")


def perform_kpca(X, n_components, kernel_type='rbf', gamma=None):
    N = X.shape[0]
    K = compute_kernel(X, X, kernel_type, gamma)
    
    # 核矩陣中心化: K_centered = K - 1_N*K - K*1_N + 1_N*K*1_N
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # 特徵分解
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # 排序並取前 k 個
    idx = np.argsort(eigenvalues)[::-1]
    alphas = eigenvectors[:, idx[:n_components]]
    lambdas = eigenvalues[idx[:n_components]]
    
    # 係數單位化: alpha = alpha / sqrt(lambda)
    # 這是為了確保在特徵空間中 ||w|| = 1
    for i in range(alphas.shape[1]):
        if lambdas[i] > 0:
            alphas[:, i] /= np.sqrt(lambdas[i])
            
    return alphas, X, lambdas # 回傳訓練集 X 用於測試時計算核

def perform_klda(X, y, n_components, kernel_type='rbf', gamma=None):
    N = X.shape[0]
    labels = np.unique(y)
    C = len(labels)
    K = compute_kernel(X, X, kernel_type, gamma)
    
    # 建構 Z 矩陣 (論文公式 11)
    Z = np.zeros((N, N))
    for label in labels:
        indices = np.where(y == label)[0]
        l_i = len(indices)
        # 在對應類別的區塊填入 1/l_i
        for r in indices:
            for c in indices:
                Z[r, c] = 1.0 / l_i
                
    # M = K Z K (Between-class)
    # N = K K (Within-class)
    M = K.dot(Z).dot(K)
    N_mat = K.dot(K)
    
    # 正則化以確保數值穩定性
    N_mat += np.eye(N) * 1e-3
    
    # 求解廣義特徵值問題 (KZK)alpha = lambda (KK)alpha
    eigenvalues, alphas = np.linalg.eig(np.linalg.pinv(N_mat).dot(M))
    
    idx = np.argsort(eigenvalues.real)[::-1]
    alphas = alphas.real[:, idx[:n_components]]
    
    return alphas, X

def testing_kernel(train_X, train_y, test_X, test_y, alphas, 
                   method_name, kernel_type='rbf', gamma=None):
    """
    alphas: 訓練得到的係數矩陣
    """
    # 1. 計算測試集與訓練集之間的核矩陣 (M_test, N_train)
    K_test = compute_kernel(test_X, train_X, kernel_type, gamma)
    K_train = compute_kernel(train_X, train_X, kernel_type, gamma)
    
    # 若是 KPCA，測試核矩陣也需要中心化
    if method_name == 'kpca':
        N_train = train_X.shape[0]
        M_test = test_X.shape[0]
        one_n_train = np.ones((N_train, N_train)) / N_train
        one_m_test = np.ones((M_test, N_train)) / N_train
        K_test = K_test - one_m_test.dot(K_train) - K_test.dot(one_n_train) + one_m_test.dot(K_train).dot(one_n_train)

    # 2. 投影到特徵空間
    test_features = np.dot(K_test, alphas)
    
    # 訓練集的投影特徵 (用於 1-NN)
    if method_name == 'kpca':
        # KPCA 的訓練特徵已在 perform_kpca 中隱含 (alphas * lambdas)
        # 但為了統一邏輯，我們直接重新計算
        K_train_centered = K_train - (np.ones((N_train, N_train))/N_train).dot(K_train) # 簡化寫法
        train_features = np.dot(K_train_centered, alphas)
    else:
        train_features = np.dot(K_train, alphas)

    # 3. 最近鄰辨識 (使用前述純 NumPy 距離公式)
    test_sq = np.sum(test_features**2, axis=1)[:, np.newaxis]
    train_sq = np.sum(train_features**2, axis=1)[np.newaxis, :]
    dists = np.sqrt(np.maximum(test_sq + train_sq - 2 * np.dot(test_features, train_features.T), 0))
    
    predictions = train_y[np.argmin(dists, axis=1)]
    accuracy = np.mean(predictions == test_y)
    print(f"Kernel {method_name.upper()} ({kernel_type}) Accuracy: {accuracy * 100:.2f}%")


def main():
    n_components = 30
    # 載入數據
    X, y = load_yale_data('Yale_Face_Database/Training')
    test_X, test_y = load_yale_data('Yale_Face_Database/Testing')

    X = X / 255.0
    test_X = test_X / 255.0
    gamma_pca = 0.0003
    gamma_lda = 0.0005

    # --- PCA 測試 ---
    X_pca, W_pca, mean_face = perform_pca(X, n_components)
    testing(X, y, test_X, test_y, W_pca, mean_face, 'pca')

    # --- LDA 測試 (Fisherface) ---
    X_lda, W_fisher = perform_lda(X, y, n_components) 
    testing(X, y, test_X, test_y, W_fisher, mean_face, 'lda')

    alphas, X, lambdas = perform_kpca(X, n_components, kernel_type='rbf', gamma=gamma_pca)
    testing_kernel(X, y, test_X, test_y, alphas, method_name='kpca', kernel_type='rbf', gamma=gamma_pca)

    alphas, X  = perform_klda(X, y, n_components, kernel_type='rbf', gamma=gamma_lda)
    testing_kernel(X, y, test_X, test_y, alphas, method_name='klda', kernel_type='rbf', gamma=gamma_lda)


if __name__ == "__main__":
    main()


