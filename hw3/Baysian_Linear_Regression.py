import numpy as np
import matplotlib.pyplot as plt # 引入 matplotlib.pyplot

# --- 1. 数据生成器 ---
def polynomial_basis_linear_model_data_generator(n, w, a, num_samples):
    x_values = np.random.uniform(-1.0, 1.0, num_samples)
    X_design = np.zeros((num_samples, n))
    for i in range(n):
        X_design[:, i] = x_values**i
    e = np.random.normal(0, np.sqrt(a), (num_samples, 1))
    y = X_design @ w + e
    return X_design, y.flatten()

# --- 2. 后验参数更新函数 ---
import numpy as np

def update_posterior(X, y, mu_0, Sigma_0, noise_variance_a):
    """
    更新贝叶斯线性回归的后验均值和协方差。

    Args:
        X (ndarray): 单个数据点的设计矩阵 (phi(x).T), 形状 (1, n)。
        y (float/ndarray): 观测值 y (标量或 (1, 1) 矩阵)。
        mu_0 (ndarray): 当前的后验均值 (n, 1)。
        Sigma_0 (ndarray): 当前的后验协方差 (n, n)。
        noise_variance_a (float): 观测噪声方差 a。
    """
    
    # 观测噪声精度 (beta = 1/a)
    beta = 1.0 / noise_variance_a
    y_matrix = np.atleast_2d(y) 
    
    # --- 1. 计算后验逆协方差 (Sigma_N_inv) ---
    # Sigma_N_inv = Sigma_0_inv + beta * X.T @ X
    Sigma_n_inv = np.linalg.inv(Sigma_0) + beta * X.T @ X
    
    # --- 2. 计算后验协方差 (Sigma_N) ---
    Sigma_n = np.linalg.inv(Sigma_n_inv)
    
    # --- 3. 计算后验均值 (mu_N) ---
    # mu_N = Sigma_N @ (Sigma_0_inv @ mu_0 + beta * X.T @ y)
    mu_n = Sigma_n @ (np.linalg.inv(Sigma_0) @ mu_0 + beta * X.T @ y_matrix) 
    
    return mu_n, Sigma_n

# --- 3. 后验预测函数 ---
def posterior_predictive(X_tilde, mu_n, Sigma_n, noise_variance_a):
    """
    计算单个新数据点 (M=1) 的后验预测均值和方差。
    
    Args:
        X_tilde (ndarray): 新数据点的设计矩阵 (phi(x).T), 形状 (1, n)。
        mu_n (ndarray): 权重的后验均值 (n, 1)。
        Sigma_n (ndarray): 权重的后验协方差 (n, n)。
        noise_variance_a (float): 观测噪声方差 a。
        
    Returns:
        tuple: (predictive_mean_scalar, predictive_variance_scalar)
    """
    
    # 1. 预测均值 (Mean): 
    # Formula: X_tilde @ mu_n
    # 形状: (1, n) @ (n, 1) -> (1, 1)
    predictive_mean = X_tilde @ mu_n
    
    # 2. 预测方差 (Variance): 
    # Formula (Total Uncertainty): 噪声方差 (a) + 模型不确定性 (X_tilde @ Sigma_n @ X_tilde.T)
    # 形状: 标量 + (1, n) @ (n, n) @ (n, 1) -> (1, 1)
    model_uncertainty = X_tilde @ Sigma_n @ X_tilde.T
    
    # 總方差 = 观測噪声方差 (a) + 模型不确定性
    predictive_variance_matrix = noise_variance_a + model_uncertainty
    
    # 返回标量值 (从 (1, 1) 矩阵中取出)
    return predictive_mean[0, 0], predictive_variance_matrix[0, 0]

# --- 4. 输出函数 ---
def print_output(mu_n, Sigma_n, predictive_mean, predictive_variance, data_count):
    # 简化输出格式以节省空间，只打印关键信息
    # 您可以根据需要调整为更详细的打印
    pass 

# --- 5. 核心绘图函数 ---
def final_plots(plot_data, n, a, w, name):
    # 定义测试 x 范围和设计矩阵
    x_test = np.linspace(-2.0, 2.0, 200) 
    X_test_design = np.array([x_test**i for i in range(n)]).T 
    y_true = X_test_design @ w 
    y_true_flat = y_true.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # --- 1. 定义四个子图的配置 ---
    # 我们将 Ground Truth 视为一个特殊的预测结果，其 mu=w, Sigma=0
    
    # 预测结果列表：(mu, Sigma, X_seen, y_seen, Title)
    plot_items = [
        # (0, 0) - Ground Truth
        {'mu': w.copy(), 'Sigma': np.zeros((n, n)), 'X_seen': np.empty((0, n)), 'y_seen': np.empty(0), 
         'title': 'Ground truth', 'row': 0, 'col': 0},
        
        # (0, 1) - Final Predict Result (使用 plot_data 中的最终数据)
        {'mu': plot_data['final']['mu'], 'Sigma': plot_data['final']['Sigma'], 
         'X_seen': plot_data['final']['X_seen'], 'y_seen': plot_data['final']['y_seen'], 
         'title': 'Predict result', 'row': 0, 'col': 1},
        
        # (1, 0) - After 10 incomes
        {'mu': plot_data['10']['mu'], 'Sigma': plot_data['10']['Sigma'], 
         'X_seen': plot_data['10']['X_seen'], 'y_seen': plot_data['10']['y_seen'], 
         'title': 'After 10 incomes', 'row': 1, 'col': 0},
        
        # (1, 1) - After 50 incomes
        {'mu': plot_data['50']['mu'], 'Sigma': plot_data['50']['Sigma'], 
         'X_seen': plot_data['50']['X_seen'], 'y_seen': plot_data['50']['y_seen'], 
         'title': 'After 50 incomes', 'row': 1, 'col': 1},
    ]

    # --- 2. 循环绘制四个子图 ---
    for item in plot_items:
        ax = axes[item['row'], item['col']]
        
        # 使用当前 mu 和 Sigma 计算预测均值和方差
        mu_n, Sigma_n = item['mu'], item['Sigma']
        X_seen, y_seen = item['X_seen'], item['y_seen']
        
        # 预测均值 (黑线)
        y_mean = (X_test_design @ mu_n).flatten()
        
        # 预测方差和标准差
        # Ground Truth 的 Sigma_n 为 0，方差只剩下 a
        y_variance = a + np.diag(X_test_design @ Sigma_n @ X_test_design.T) 
        y_std = np.sqrt(y_variance)

        # 绘制预测均值 (黑线)
        ax.plot(x_test, y_mean, color='black', label="Predictive Mean")
        
        # 绘制 ±1 标准差 (红线)
        ax.plot(x_test, y_mean + y_std, color='red', linestyle='-', linewidth=1) 
        ax.plot(x_test, y_mean - y_std, color='red', linestyle='-', linewidth=1)
        
        # 绘制已观测数据点 (Ground Truth 不绘制数据点)
        if item['title'] != 'Ground truth' and len(X_seen) > 0:
            ax.scatter(X_seen[:, 1], y_seen, color='blue', s=10, label="Observed Data") 
            
        ax.set_title(item['title'])
        ax.set_xlim([-2, 2])
        ax.set_ylim([-20, 20])
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{name}.png")


# --- 6. 示例运行 (Example usage) 主逻辑 ---
if __name__ == '__main__':
    # Parameters
    name = "Case3"
    b = 1
    n = 3 
    a = 3  
    w = np.array([1, 2, 3]).reshape(-1, 1) # True weights (n, 1)
    num_samples = 100 # 确保我们至少有 50 个点
    
    # 设置随机种子以保证结果可复现 (可选)
    # np.random.seed(42)

    # Generate all data points
    X_full, y_full = polynomial_basis_linear_model_data_generator(n, w, a, num_samples)
    
    # Prior parameters
    mu = np.zeros((n, 1))
    Sigma = np.eye(n) / b
    
    plot_data = {}

    # 迭代更新后验参数
    for i in range(len(X_full)):
        
        X_batch = X_full[i].reshape(1, -1) 
        y_batch = y_full[i]
        
        # 更新后验参数
        mu, Sigma = update_posterior(X_batch, y_batch, mu, Sigma, a)
        
        # 记录 i=10 和 i=50 时的后验参数
        current_count = i + 1
        
        if current_count == 10:
            plot_data['10'] = {'mu': mu.copy(), 'Sigma': Sigma.copy(), 'X_seen': X_full[:current_count], 'y_seen': y_full[:current_count]}
            print(f"--- Saved parameters after {current_count} points ---")
            
        if current_count == 50:
            plot_data['50'] = {'mu': mu.copy(), 'Sigma': Sigma.copy(), 'X_seen': X_full[:current_count], 'y_seen': y_full[:current_count]}
            print(f"--- Saved parameters after {current_count} points ---")
            
    plot_data['final'] = {'mu': mu.copy(), 'Sigma': Sigma.copy(), 'X_seen': X_full[:current_count], 'y_seen': y_full[:current_count]}
    final_plots(plot_data, n, a, w, name)
