import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def Hbeta(D=np.array([]), beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P


def pca(X=np.array([]), no_dims=50):
    print("Preprocessing data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def save_gif(frames, filename):
    print(f"Saving GIF to {filename}...")
    imageio.mimsave(filename, frames, fps=10)

def plot_similarity(P, Q, title):
    plt.figure(figsize=(12, 4))
    p_flat = P[P > 1e-12].flatten()
    q_flat = Q[Q > 1e-12].flatten()
    
    plt.subplot(1, 2, 1)
    plt.hist(np.log10(p_flat), bins=50, color='blue', alpha=0.6)
    plt.title(f"{title}: High-dim P (Log10)")
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log10(q_flat), bins=50, color='red', alpha=0.6)
    plt.title(f"{title}: Low-dim Q (Log10)")
    plt.savefig(f"similarity_{title.lower()}.png")
    plt.show()


def run_embedding(X, labels, method='tsne', no_dims=2, perplexity=30.0):
    X = pca(X, 50).real
    (n, d) = X.shape
    max_iter = 500 
    Y = np.random.randn(n, no_dims)
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    frames = []

    P = x2p(X, 1e-5, perplexity)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P * 4, 1e-12) # Early exaggeration

    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), 1)
        dist = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
        
        if method == 'tsne':
            num = 1. / (1. + dist) # t-分布
        else:
            num = np.exp(-dist)    # 高斯分布 (Symmetric SNE)
            
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = P - Q
        dY = np.zeros((n, no_dims))
        if method == 'tsne':
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        else:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Update
        momentum = 0.5 if iter < 20 else 0.8
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < 0.01] = 0.01
        iY = momentum * iY - 500 * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, 0)

        if iter == 100: P = P / 4. 

        if iter % 10 == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(Y[:, 0], Y[:, 1], 10, labels, cmap='tab10')
            ax.set_title(f"{method.upper()} Iteration {iter}")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frames.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
            plt.close()

    save_gif(frames, f"{method}_optimization.gif")
    return Y, P, Q


if __name__ == "__main__":
    X = np.loadtxt("tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("tsne_python/mnist2500_labels.txt")
    perplexitys = [10, 30, 50]

    for perplexity in perplexitys:
        print(f"-------------- perplexity = {perplexity} --------------------------------------------------------")
        Y_tsne, P_tsne, Q_tsne = run_embedding(X, labels, method='tsne', perplexity=perplexity)
        plot_similarity(P_tsne, Q_tsne, f"t-SNE-perplexity{perplexity}")

        Y_ssne, P_ssne, Q_ssne = run_embedding(X, labels, method='ssne', perplexity=perplexity)
        plot_similarity(P_ssne, Q_ssne, f"Symmetric-SNE-perplexity{perplexity}")
