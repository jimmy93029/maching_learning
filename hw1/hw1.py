import numpy as np
from utils import read_input_file, data_matrix, gard_and_hessian, draw, print_fitting_line, inverse
import matplotlib.pyplot as plt


def closed_form_LSE(A, b, regulization):

    x = inverse(A.T @ A + regulization) @ A.T @ b

    return x


def steepest_descent_method(A, b, _lambda, x):

    rate = 0.0001
    tolerance = 1e-6

    for i in range(5000):
        grad_f = 2 * A.T @ A @ x - 2 * A.T @ b + _lambda * np.sign(x)

        x_new = x - rate * grad_f

        if np.linalg.norm(x_new - x) < tolerance:
            # print(f"Converged at iteration {i}")
            break

        x = x_new

    return x


def newton_method(A, b, x):

    grad, hessian = gard_and_hessian(A, b, x)

    x = x - inverse(hessian) @ grad

    return x


def Case(x_data, y_data, degree, _lambda, filename):

    A = data_matrix(x_data, degree)
    b = y_data.reshape(-1, 1)  # turn y_data.shape (1, n) to b (n, 1)

    x1 = closed_form_LSE(A, b, regulization=_lambda * np.eye(degree, dtype=float))
    x2 = steepest_descent_method(A, b, _lambda, x=np.eye(degree, 1, dtype=float))
    x3 = newton_method(A, b, x=np.eye(degree, 1, dtype=float))

    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    error1 = draw(x1, x_data, y_data, axs[0])
    error2 = draw(x2, x_data, y_data, axs[1])
    error3 = draw(x3, x_data, y_data, axs[2])

    print("LSE")
    print(f"Fitting line: {print_fitting_line(x1)}")
    print(f"Total error: {error1}")
    print()
    print("Steepest descent")
    print(f"Fitting line: {print_fitting_line(x2)}")
    print(f"Total error: {error2}")
    print()
    print("Newton's method")
    print(f"Fitting line: {print_fitting_line(x3)}")
    print(f"Total error: {error3}")
    print()

    fig.savefig(filename, format='png')
    plt.close(fig) 


def main():
    x_data, y_data = read_input_file('input.txt')

    Case(x_data, y_data, degree=2, _lambda=0, filename="Case1.png")
    Case(x_data, y_data, degree=3, _lambda=0, filename="Case2.png")
    Case(x_data, y_data, degree=3, _lambda=10000, filename="Case3.png")


if __name__ == '__main__':
    main()

