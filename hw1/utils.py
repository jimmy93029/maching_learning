import numpy as np
import torch
import torch.nn as nn


np.set_printoptions(precision=11)

def read_input_file(filename):
    # Initialize an empty list to store the data
    x_data, y_data = [], []
    
    # Open the file and read each line
    with open(filename, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split by comma
            values = line.strip().split(',')
            # Convert the values to floats and append to the data list
            x_data.append(float(values[0]))
            y_data.append(float(values[1]))

    # Convert the list of data into a numpy array
    return np.array(x_data), np.array(y_data)


def data_matrix(x_data, degree):
    A = np.vstack([x_data**i for i in range(degree)]).T
    return A


# Gauss-Jordan elimintation
def inverse(X):
    n = X.shape[0]
    Y = np.copy(X)
    
    I = np.eye(n)
    
    # Augment the matrix X with the identity matrix I
    Augmented = np.hstack((Y, I))

    for i in range(n):
        # Find the pivot row and swap if necessary to avoid division by zero
        if Augmented[i, i] == 0:
            for k in range(i+1, n):
                if Augmented[k, i] != 0:
                    Augmented[[i, k]] = Augmented[[k, i]]  # Swap rows
                    break
            else:
                print("Matrix is singular and cannot be inverted")
                return None
        
        # Normalize the pivot row
        Augmented[i] = Augmented[i] / Augmented[i, i]
        
        # Eliminate the column entries below and above the pivot
        for j in range(n):
            if i != j:
                Augmented[j] -= Augmented[j, i] * Augmented[i]
    
    # The right half of the augmented matrix is the inverse of X
    inverse_matrix = Augmented[:, n:]
    return inverse_matrix


def gard_and_hessian(A, b, x):
    A_tensor = torch.tensor(A, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    
    # Compute Residual and grad
    residual = A_tensor @ x_tensor - b_tensor
    loss = torch.norm(residual) ** 2
    grad = torch.autograd.grad(loss, x_tensor, create_graph=True)[0]
    
    # Compute Hessian
    n = len(x_tensor)
    hessian = torch.zeros(n, n)
    
    for i in range(n):
        grad_grad = torch.autograd.grad(grad[i], x_tensor, retain_graph=True)[0]
        grad_grad = grad_grad.squeeze()
        hessian[i] = grad_grad
        
    return grad.detach().numpy(), hessian.detach().numpy()


def draw(x, x_data, y_data, ax):

    x = x[::-1]

    poly = np.poly1d(x)

    y_fit = poly(x_data)

    ax.scatter(x_data, y_data, color='red', label='Data points')
    ax.plot(x_data, y_fit, label=f'Fitting curve', color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    error = np.sum((y_data - y_fit)**2)

    return error


def print_fitting_line(x):
    n = len(x) - 1  
    terms = []
    
    for i, coef in enumerate(x[::-1]): 
        power = n - i  
        
        if power == 1:
            terms.append(f"{coef:.6f} * x")  
        elif power == 0:
            terms.append(f"{coef:.6f}")  
        else:
            terms.append(f"{coef:.6f} * x^{power}")  
    
    return " + ".join(terms)


