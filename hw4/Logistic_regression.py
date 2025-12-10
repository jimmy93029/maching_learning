import numpy as np
import matplotlib.pyplot as plt


# 1. Data Generation (You need your HW3 Gaussian generator here) [cite: 7, 8, 9]
# D1: Cluster 1 (Label y=1)
# D2: Cluster 2 (Label y=0) - Using 0 and 1 for labels is typical for log-reg
def generate_data(n, mx, my, vx, vy):
    # Use your Gaussian random number generator here
    x = np.random.normal(mx, np.sqrt(vx), n) 
    y = np.random.normal(my, np.sqrt(vy), n) 
    return np.array([x, y]).T

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_h(X, w):
    # X @ w is the linear combination (z = w^T * x)
    return sigmoid(X @ w)

def compute_gradient(X, y, h, m):
    # Gradient vector (3 x 1)
    return (1 / m) * X.T @ (h - y)

def compute_hessian(X, h, m):
    # Diagonal matrix R with elements h * (1 - h)
    R = np.diag(h * (1 - h))
    # Hessian matrix (3 x 3)
    return (1 / m) * X.T @ R @ X

def steepest_gradient_descent(X, y, m, alpha=0.01, max_iter=50000, tol=1e-6):
    w = np.zeros(X.shape[1]) # Initialize weights (w0, w1, w2)
    
    for i in range(max_iter):
        h = compute_h(X, w)
        grad = compute_gradient(X, y, h, m)
        
        # Update rule
        w_new = w - alpha * grad
        
        # Check for convergence (e.g., small change in w) [cite: 12]
        if np.linalg.norm(w - w_new) < tol:
            print(f"Gradient Descent converged after {i+1} iterations.")
            break
            
        w = w_new
        
    return w

def newtons_method(X, y, m, alpha_gd=0.01, max_iter=100000, tol=1e-5):
    w = np.zeros(X.shape[1]) # Initialize weights
    
    for i in range(max_iter):
        h = compute_h(X, w)
        grad = compute_gradient(X, y, h, m)
        H = compute_hessian(X, h, m)
        
        try:
            # Attempt to use Newton's step (requires inverse of H)
            H_inv = np.linalg.inv(H)
            update_step = H_inv @ grad
            w_new = w - update_step
            method = "Newton"
            print(method)
        
        except np.linalg.LinAlgError:
            # Fallback: Hessian is singular, use steepest descent instead [cite: 11]
            # Use the pre-calculated gradient
            update_step = alpha_gd * grad 
            w_new = w - update_step
            method = "Steepest Descent Fallback"
            print(method)
            
        # Check for convergence [cite: 12]
        if np.linalg.norm(w - w_new) < tol:
            print(f"{method} converged after {i+1} iterations.")
            break
            
        w = w_new

    return w

def predict(X, w):
    # Predict probabilities, then classify based on a threshold (0.5)
    probabilities = compute_h(X, w)
    # Cluster 1 (1) if prob > 0.5, Cluster 2 (0) otherwise
    return (probabilities >= 0.5).astype(int)

def evaluate(y_true, y_pred):
    # Calculate True Positives (TP), False Positives (FP), etc.
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    # Confusion Matrix [cite: 14]
    confusion_matrix = np.array([[TP, FN], [FP, TN]])
    
    # Sensitivity (Successfully predict cluster 1) [cite: 14, 35]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity (Successfully predict cluster 2) [cite: 14, 36]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return confusion_matrix, sensitivity, specificity

# --- 6. Visualization ---
def plot_decision_boundary(ax, X, y, w, title, color_map):
    # Plot the scatter data
    ax.scatter(X[y == 1, 1], X[y == 1, 2], c='red', marker='o', label='Cluster 1 (D1)')
    ax.scatter(X[y == 0, 1], X[y == 0, 2], c='blue', marker='x', label='Cluster 2 (D2)')

    # Calculate the decision boundary line: w0 + w1*x + w2*y = 0
    # Line is y = (-w[0] - w[1]*x) / w[2]
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_line = np.linspace(x_min, x_max, 100)
    
    # Check if w[2] is close to zero (vertical line case)
    if np.abs(w[2]) > 1e-6:
        y_line = (-w[0] - w[1] * x_line) / w[2]
        ax.plot(x_line, y_line, color='black', linestyle='-', linewidth=2, label='Boundary')
    else:
        # Vertical boundary case: x = -w[0] / w[1]
        x_val = -w[0] / w[1]
        ax.axvline(x=x_val, color='black', linestyle='-', linewidth=2, label='Boundary')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)
    ax.grid(True)

def Case(n, mx1, my1, vx1, vy1, mx2, my2, vx2, vy2, name="Case1"):

    D1_data = generate_data(n, mx1, my1, vx1, vy1)
    D2_data = generate_data(n, mx2, my2, vx2, vy2)

    # 2. Preprocessing: Combine and create design matrix X [cite: 8, 9]
    # X contains [1, x, y] for all data points
    X1 = np.c_[np.ones(n), D1_data]
    X2 = np.c_[np.ones(n), D2_data]
    X = np.vstack((X1, X2)) # Combined data (2n x 3)

    # Labels: 1 for D1 (Cluster 1), 0 for D2 (Cluster 2)
    y1 = np.ones(n)
    y2 = np.zeros(n)
    y = np.concatenate((y1, y2)) # Combined labels (2n x 1)

    m, d = X.shape # m = 100, d = 3

    # --- Run Gradient Descent ---
    w_gd = steepest_gradient_descent(X, y, m, alpha=0.01)
    y_pred_gd = predict(X, w_gd)
    cm_gd, sen_gd, spec_gd = evaluate(y, y_pred_gd)

    # --- Run Newton's Method ---
    w_newton = newtons_method(X, y, m)
    y_pred_newton = predict(X, w_newton)
    cm_newton, sen_newton, spec_newton = evaluate(y, y_pred_newton)

    # Print Gradient Descent Results (for completeness/comparison) [cite: 809-823]
    print("Gradient descent:")
    print("W:")
    print(f"{w_gd[0]:.10f}")
    print(f"{w_gd[1]:.10f}")
    print(f"{w_gd[2]:.10f}")

    print("\nConfusion Matrix:")
    print(" " * 19 + "Predict cluster 1" + " Predict cluster 2")
    print("Is cluster 1" + f"{cm_gd[0, 0]:>18}" + f"{cm_gd[0, 1]:>17}")
    print("Is cluster 2" + f"{cm_gd[1, 0]:>18}" + f"{cm_gd[1, 1]:>17}")
    print(f"Sensitivity (Successfully predict cluster 1): {sen_gd:.5f}")
    print(f"Specificity (Successfully predict cluster 2): {spec_gd:.5f}")

    # Print Newton's Method Results (The main target of this request) [cite: 824-836]
    print("\n" + "="*40)
    print("Newton's method:")
    print("W:")
    print(f"{w_newton[0]:.10f}")
    print(f"{w_newton[1]:.10f}")
    print(f"{w_newton[2]:.10f}")

    print("\nConfusion Matrix:")
    print(" " * 19 + "Predict cluster 1" + " Predict cluster 2")
    print("Is cluster 1" + f"{cm_newton[0, 0]:>18}" + f"{cm_newton[0, 1]:>17}")
    print("Is cluster 2" + f"{cm_newton[1, 0]:>18}" + f"{cm_newton[1, 1]:>17}")
    print(f"Sensitivity (Successfully predict cluster 1): {sen_newton:.5f}")
    print(f"Specificity (Successfully predict cluster 2): {spec_newton:.5f}")

    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Logistic Regression Results for Case 1', fontsize=16)

    axes[0].scatter(D1_data[:, 0], D1_data[:, 1], c='red', marker='o', label='Cluster 1 (D1)')
    axes[0].scatter(D2_data[:, 0], D2_data[:, 1], c='blue', marker='x', label='Cluster 2 (D2)')
    axes[0].set_title('Ground Truth')
    axes[0].grid(True)
    axes[0].legend()

    # 2. Plot Gradient Descent Prediction [cite: 804, 805]
    plot_decision_boundary(axes[1], X, y, w_gd, 'Gradient Descent Prediction', 'RdBu')

    # 3. Plot Newton's Method Prediction [cite: 804, 806]
    plot_decision_boundary(axes[2], X, y, w_newton, 'Newton\'s Method Prediction', 'RdBu')

    # 3. Plot Newton's Method Prediction [cite: 804, 806]
    plot_decision_boundary(axes[2], X, y, w_newton, 'Newton\'s Method Prediction', 'RdBu')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(name)



if __name__ == "__main__":
    # Parameters for Case 1 
    Case(n=50, mx1=1, my1=1, vx1=2, vy1=2, mx2=10, my2=10, vx2=2, vy2=2, name="Case1")
    # parameters for Case 2
    Case(n=50, mx1=1, my1=1, vx1=2, vy1=2, mx2=3, my2=3, vx2=4, vy2=4, name="Case2")
