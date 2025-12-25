import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import cv2
import shutil
import os


def load_yale_data(folder_path):
    images = []
    labels = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return np.array([]), np.array([])
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pgm"):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
                subject_id = int(filename.replace("subject", "").split(".")[0])
                labels.append(subject_id)
                
    return np.array(images, dtype=np.float64), np.array(labels)


def perform_pca(X, n_components):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    
    # Gram matrix
    gram_matrix = np.dot(X_centered, X_centered.T)
    eigenvalues, eigenvectors_small = np.linalg.eigh(gram_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_small = eigenvectors_small[:, idx]
    
    # Project to high dimension space
    W_pca = np.dot(X_centered.T, eigenvectors_small[:, :n_components])
    
    # unitlization
    for i in range(W_pca.shape[1]):
        W_pca[:, i] /= np.linalg.norm(W_pca[:, i])
        
    X_pca = np.dot(X_centered, W_pca)
    return X_pca, W_pca, mean_face


def perform_lda(X, y, n_components):
    n_samples, n_features = X.shape
    class_labels = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    X_centered = X - mean_overall
    
    Kw = np.zeros((n_samples, n_samples))
    Kb = np.zeros((n_samples, n_samples))
    
    for c in class_labels:
        idx_c = (y == c)
        X_c = X[idx_c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        
        v_b = np.dot(X_centered, (mean_c - mean_overall).T).reshape(-1, 1)
        Kb += n_c * np.dot(v_b, v_b.T)
        
        v_w = np.dot(X_centered, (X_c - mean_c).T)
        Kw += np.dot(v_w, v_w.T)

    Kw += np.eye(n_samples) * 1e-6
    A = np.dot(np.linalg.pinv(Kw), Kb)
    eigenvalues, alphas = np.linalg.eig(A)
    
    idx = np.argsort(eigenvalues.real)[::-1]
    alphas = alphas.real[:, idx][:, :n_components]
    W_lda = np.dot(X_centered.T, alphas)
    
    for i in range(W_lda.shape[1]):
        W_lda[:, i] /= np.linalg.norm(W_lda[:, i])
    
    X_lda = np.dot(X_centered, W_lda)
    return X_lda, W_lda


def compute_kernel(X1, X2, kernel_type='linear', gamma=None, degree=3, coef0=1):
    if gamma is None:
        gamma = 1.0 / X1.shape[1]

    if kernel_type == 'linear':
        return np.dot(X1, X2.T)
    elif kernel_type == 'rbf':
        dists_sq = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-gamma * dists_sq)
    elif kernel_type == 'poly':
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree
    elif kernel_type == 'sigmoid':
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def perform_kpca(X, n_components, kernel_type='rbf', gamma=None):
    N = X.shape[0]
    K = compute_kernel(X, X, kernel_type, gamma)
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    idx = np.argsort(eigenvalues)[::-1]
    alphas = eigenvectors[:, idx[:n_components]]
    lambdas = eigenvalues[idx[:n_components]]
    
    for i in range(alphas.shape[1]):
        if lambdas[i] > 1e-10:
            alphas[:, i] /= np.sqrt(lambdas[i])
            
    return alphas, lambdas


def perform_klda(X, y, n_components, kernel_type='rbf', gamma=None):
    N = X.shape[0]
    labels = np.unique(y)
    K = compute_kernel(X, X, kernel_type, gamma)
    
    Z = np.zeros((N, N))
    for label in labels:
        indices = np.where(y == label)[0]
        Z[np.ix_(indices, indices)] = 1.0 / len(indices)
                
    M = K.dot(Z).dot(K)
    N_mat = K.dot(K) + np.eye(N) * 1e-3
    
    eigenvalues, alphas = np.linalg.eig(np.linalg.pinv(N_mat).dot(M))
    idx = np.argsort(eigenvalues.real)[::-1]
    alphas = alphas.real[:, idx[:n_components]]
    
    return alphas


def knn_predict(train_features, train_y, test_features, k=3):
    dists = cdist(test_features, train_features, 'euclidean')
    
    predictions = []
    closest_indices = [] 
    
    for i in range(len(test_features)):
        k_nearest_indices = np.argsort(dists[i])[:k]
        closest_indices.append(k_nearest_indices[0]) 
        
        k_nearest_labels = train_y[k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
        
    return np.array(predictions), np.array(closest_indices)


def testing(train_X, train_y, test_X, test_y, W, mean_face, method_name, k=3, img_shape=(231, 195)):
    recog_dir = f'experiments/{method_name}/recognition/'
    recon_dir = f'experiments/{method_name}/reconstruction/'
    for d in [recog_dir, recon_dir]:
        if not os.path.exists(d): os.makedirs(d)

    train_features = np.dot(train_X - mean_face, W)
    test_features = np.dot(test_X - mean_face, W)

    predictions, closest_idx = knn_predict(train_features, train_y, test_features, k=k)
    accuracy = np.mean(predictions == test_y)
    print(f"[{method_name.upper()}] Accuracy: {accuracy * 100:.2f}%")

    test_X_reconstructed = np.dot(test_features, W.T) + mean_face

    sample_idx = np.random.choice(len(test_X), min(10, len(test_X)), replace=False)
    for i in sample_idx:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.imshow(test_X[i].reshape(img_shape), cmap='gray'); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(train_X[closest_idx[i]].reshape(img_shape), cmap='gray'); plt.axis('off')
        plt.savefig(os.path.join(recog_dir, f"recog_{i}.png")); plt.close()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.imshow(test_X[i].reshape(img_shape), cmap='gray'); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(test_X_reconstructed[i].reshape(img_shape), cmap='gray'); plt.axis('off')
        plt.savefig(os.path.join(recon_dir, f"recon_{i}.png")); plt.close()


def testing_kernel(train_X, train_y, test_X, test_y, alphas, method_name, k=3, 
                   kernel_type='rbf', gamma=None, img_shape=(231, 195)):
    base_dir = f'experiments/{method_name}_{kernel_type}'
    recog_dir = os.path.join(base_dir, 'recognition')
    if not os.path.exists(recog_dir): os.makedirs(recog_dir)

    K_test = compute_kernel(test_X, train_X, kernel_type, gamma)
    K_train = compute_kernel(train_X, train_X, kernel_type, gamma)
    
    if method_name == 'kpca':
        N = train_X.shape[0]
        one_n = np.ones((N, N)) / N
        M = test_X.shape[0]
        one_m = np.ones((M, N)) / N
        K_test = K_test - one_m.dot(K_train) - K_test.dot(one_n) + one_m.dot(K_train).dot(one_n)
        K_train = K_train - one_n.dot(K_train) - K_train.dot(one_n) + one_n.dot(K_train).dot(one_n)

    test_features = np.dot(K_test, alphas)
    train_features = np.dot(K_train, alphas)

    predictions, closest_idx = knn_predict(train_features, train_y, test_features, k=k)
    accuracy = np.mean(predictions == test_y)
    print(f"[{method_name.upper()}] Kernel: {kernel_type}, Accuracy: {accuracy * 100:.2f}%")


def main():

    results_path = 'experiments'
    
    if os.path.exists(results_path):
        print(f"Cleaning up old results in {results_path}...")
        shutil.rmtree(results_path)
    
    os.makedirs(results_path)

    n_components = 30
    k_val = 3
    img_shape = (231, 195)
    
    X, y = load_yale_data('Yale_Face_Database/Training')
    test_X, test_y = load_yale_data('Yale_Face_Database/Testing')
    X /= 255.0; test_X /= 255.0

    # 1. Linear Methods
    X_pca, W_pca, mean_face = perform_pca(X, n_components)
    testing(X, y, test_X, test_y, W_pca, mean_face, 'pca', k=k_val, img_shape=img_shape)

    _, W_fisher = perform_lda(X, y, n_components) 
    testing(X, y, test_X, test_y, W_fisher, mean_face, 'lda', k=k_val, img_shape=img_shape)

    # 2. Kernel Methods Experiment
    kernel_configs = [
        {'type': 'rbf', 'gamma': 0.0003},
        {'type': 'poly', 'gamma': 0.1, 'degree': 3},
        {'type': 'sigmoid', 'gamma': 0.0001}
    ]

    for config in kernel_configs:
        kt = config['type']
        ga = config['gamma']
        
        # KPCA
        alphas_kpca, _ = perform_kpca(X, n_components, kernel_type=kt, gamma=ga)
        testing_kernel(X, y, test_X, test_y, alphas_kpca, 'kpca', k=k_val, kernel_type=kt, gamma=ga)

        # KLDA
        alphas_klda = perform_klda(X, y, n_components, kernel_type=kt, gamma=ga)
        testing_kernel(X, y, test_X, test_y, alphas_klda, 'klda', k=k_val, kernel_type=kt, gamma=ga)


if __name__ == '__main__':
    main()