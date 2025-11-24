import numpy as np
import csv
import sys
from libsvm.svmutil import svm_train, svm_predict
from libsvm.svm import *
from scipy.spatial.distance import cdist # For custom kernel (Task 3)


# --- Placeholder/Simulated LIBSVM Interface ---

def load_data(filepath_X, filepath_Y):
    # Loads data and labels from CSV files.
    try:
        X = np.loadtxt(filepath_X, delimiter=',')
        Y = np.loadtxt(filepath_Y, delimiter=',')
        # Ensure Y is in (N,) shape
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        elif Y.ndim != 1:
            raise ValueError("Y_data has unexpected shape.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    return X, Y


def train_and_predict_svm(X_train, Y_train, X_test, Y_test, kernel_type='linear', C=1.0, gamma='auto'):
    # Simulates SVM training and prediction using LIBSVM (or Scikit-learn SVC).
    print(f"  > Training with Kernel: {kernel_type}, C={C}, gamma={gamma}")

    # 1. Build LIBSVM parameter string
    param_list = ['-s 0'] # -s 0: C-SVC

    # Core: Set kernel type (-t) and related parameters
    if kernel_type == 'linear':
        param_list.append('-t 0')
    elif kernel_type == 'poly':
        param_list.append('-t 1 -d 3') # Default degree 3 (-d 3)
    elif kernel_type == 'rbf':
        param_list.append('-t 2')
        param_list.append(f'-g {gamma}')
    elif kernel_type == 'precomputed':
        param_list.append('-t 4')

    param_list.append(f'-c {C}')
    param_list.append('-q') # Quiet mode

    param_str = ' '.join(param_list)

    # 2. train model
    model = svm_train(Y_train, X_train, param_str)

    # 3. Predict and get accuracy
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)

    accuracy = p_acc[0]
    return accuracy


def comprehensive_grid_search(X, Y, folds=5):
    # Systematically searches (Grid Search) C, Gamma, and Degree for Linear, Poly, and RBF kernels.
    
    # 1. Define hyperparameter search ranges
    C_range = [0.1, 1, 10, 100]        # Penalty parameter C
    Gamma_range = [0.0001, 0.001, 0.01, 0.1] # Gamma for RBF and Poly kernels
    Degree_range = [2, 3]              # Degree d for Polynomial kernel

    overall_best_acc = -1.0
    overall_best_config = {}

    print("================== Starting Comprehensive Kernel Grid Search ================")

    # --- 2. Linear Kernel (-t 0): Search C only ---
    print("\n--- Searching Linear Kernel (-t 0) ---")

    for C in C_range:
        param_str = f'-s 0 -t 0 -c {C} -v {folds} -q'

        # Performs K-fold cross-validation
        cv_acc = svm_train(Y, X, param_str) 

        print(f"Linear: C={C:.2e} -> CV Acc = {cv_acc:.4f}%")
        config = {'kernel': 'linear', 'C': C, 'acc': cv_acc}

        if cv_acc > overall_best_acc:
            overall_best_acc = cv_acc
            overall_best_config = config


    # --- 3. RBF Kernel (-t 2): Search C x Gamma ---
    print("\n--- Searching RBF Kernel (-t 2) ---")

    for C in C_range:
        for gamma in Gamma_range:
            param_str = f'-s 0 -t 2 -c {C} -g {gamma} -v {folds} -q'

            # Performs K-fold cross-validation
            cv_acc = svm_train(Y, X, param_str)

            print(f"RBF: C={C:.2e}, g={gamma:.2e} -> CV Acc = {cv_acc:.4f}%")
            config = {'kernel': 'rbf', 'C': C, 'gamma': gamma, 'acc': cv_acc}

            if cv_acc > overall_best_acc:
                overall_best_acc = cv_acc
                overall_best_config = config


    # --- 4. Polynomial Kernel (-t 1): Search C x Degree x Gamma ---
    print("\n--- Searching Polynomial Kernel (-t 1) (coef0 r=0) ---")

    for C in C_range:
        for degree in Degree_range:
            for gamma in Gamma_range:
                param_str = f'-s 0 -t 1 -c {C} -d {degree} -g {gamma} -r 0 -v {folds} -q'

                # Performs K-fold cross-validation
                cv_acc = svm_train(Y, X, param_str)

                print(f"Poly: C={C:.2e}, d={degree}, g={gamma:.2e} -> CV Acc = {cv_acc:.4f}%")
                config = {'kernel': 'poly', 'C': C, 'degree': degree, 'gamma': gamma, 'acc': cv_acc}

                if cv_acc > overall_best_acc:
                    overall_best_acc = cv_acc
                    overall_best_config = config


    # 5. Final summary
    print("\n================== Grid Search Final Summary ==================")
    print(f"Best Kernel Found: {overall_best_config.get('kernel').upper()}")
    print(f"Best CV Accuracy: {overall_best_acc:.4f}%")

    # Return only the essential parameters
    final_params = {
        'kernel': overall_best_config.get('kernel'),
        'C': overall_best_config.get('C'),
        'gamma': overall_best_config.get('gamma'),
        'degree': overall_best_config.get('degree', None)
    }

    return final_params


# --- Task 3: Custom Kernel K(x_i, x_j) ---

def linear_rbf_kernel_matrix(X1, X2, C_rbf, gamma):
    # Computes the custom kernel matrix: K_new = K_linear + C_rbf * K_RBF.

    # 1. Compute Linear Kernel: K_linear = X1 * X2.T
    K_linear = X1 @ X2.T

    # 2. Compute RBF Kernel: K_RBF = exp(-gamma * ||x_i - x_j||^2)
    dist_sq = cdist(X1, X2, metric='sqeuclidean')
    K_RBF = np.exp(-gamma * dist_sq)

    # 3. Combine the new kernel function
    K_new = K_linear + C_rbf * K_RBF

    return K_new

# --- Task 3 Core LIBSVM Interface Function ---

def train_and_predict_custom_svm(X_train, Y_train, X_test, Y_test, C, gamma, C_rbf=1.0):
    # Trains and predicts an SVM using the precomputed custom kernel (-t 4).

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    print(f"  > Training with Custom Kernel (Linear + RBF). C={C}, gamma={gamma}, C_rbf={C_rbf}")

    # 1 Train Data Formatting
    K_train_new = linear_rbf_kernel_matrix(X_train, X_train, C_rbf, gamma)

    # 2 Train Model
    param_str = f'-s 0 -t 4 -c {C} -q'
    model = svm_train(Y_train, K_train_new, param_str)

    # 3 Test Data
    K_test_new = linear_rbf_kernel_matrix(X_test, X_train, C_rbf, gamma)

    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)

    accuracy = p_acc[0]

    return accuracy


# --- Main Execution Function ---

def main():
    # --- 0. Data Loading ---
    X_train, Y_train = load_data('X_train.csv', 'Y_train.csv')
    X_test, Y_test = load_data('X_test.csv', 'Y_test.csv')

    if X_train is None:
        print("Data loading failed. Exiting.")
        return

    # --- Task 1: Kernel Comparison ---
    print("====================== Task 1: Kernel Comparison ======================")
    results_task1 = {}

    # Use a fixed C value for baseline comparison
    base_C = 1.0

    # Linear Kernel (t=0)
    acc_linear = train_and_predict_svm(X_train, Y_train, X_test, Y_test, kernel_type='linear', C=base_C)
    results_task1['Linear'] = acc_linear

    # Polynomial Kernel (t=1)
    acc_poly = train_and_predict_svm(X_train, Y_train, X_test, Y_test, kernel_type='poly', C=base_C)
    results_task1['Polynomial'] = acc_poly

    # RBF Kernel (t=2) - Initial attempt with fixed gamma
    base_gamma = 0.001
    acc_rbf = train_and_predict_svm(X_train, Y_train, X_test, Y_test, kernel_type='rbf', C=base_C, gamma=base_gamma)
    results_task1['RBF (Fixed Params)'] = acc_rbf

    print("\n--- Task 1 Summary (C=1.0, Gamma=0.001) ---")
    for k, v in results_task1.items():
        print(f"  {k}: {v:.4f}% Accuracy")


    # --- Task 2: Grid Search for Best RBF Params (C, Gamma) ---
    print("\n================ Task 2: Grid Search for Best RBF Params ================")

    # Execute comprehensive grid search
    best_config = comprehensive_grid_search(X_train, Y_train, folds=5)

    # Task 2 best parameters (RBF)
    best_C_rbf = best_config.get('C')
    best_gamma_rbf = best_config.get('gamma')

    # Final evaluation using optimal RBF parameters
    acc_rbf_optimized = train_and_predict_svm(X_train, Y_train, X_test, Y_test, kernel_type='rbf', C=best_C_rbf, gamma=best_gamma_rbf)

    print(best_config)
    print(f"acc_rbf_optimized = {acc_rbf_optimized}")


    # --- Task 3: Custom Kernel (Linear + RBF) ---
    print("\n================== Task 3: Custom Kernel (Linear + RBF) ==================")

    # Use optimal C and gamma from Task 2, and a default C_rbf
    best_C = 10
    best_gamma = 0.01
    custom_C_rbf = 1.0 # This should ideally be optimized via grid search as well

    acc_custom_kernel = train_and_predict_custom_svm(
        X_train, Y_train,
        X_test, Y_test,
        C=best_C,
        gamma=best_gamma,
        C_rbf=custom_C_rbf
    )
    print(f"Custom Kernel (Linear + RBF) Accuracy: {acc_custom_kernel:.4f}%")


# Execute main function
if __name__ == "__main__":
    main()