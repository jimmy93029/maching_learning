import math


def binomial_likelihood(n, k, p):
    """ Calculate binomial likelihood for k successes out of n trials given probability p """
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def update_posterior(prior_a, prior_b, data):
    """ Update the Beta posterior based on the data (0s and 1s) """
    k = data.count('1')  # Number of successes (1s)
    n = len(data)  # Total number of trials
    zero_count = n - k  # Number of failures (0s)

    # MLE of Binomial distribution
    p_mle = k / n

    # Binomial likelihood
    likelihood = binomial_likelihood(n, k, p_mle)

    # Update Beta posterior parameters
    posterior_a = prior_a + k
    posterior_b = prior_b + zero_count

    return likelihood, posterior_a, posterior_b


def Case(trials, a, b):
    for i in range(len(trials)):
        trial = trials[i]
        print(f"case {i+1}: {trial}")
        a0, b0 = a, b
        likelihood, a, b = update_posterior(a0, b0, trial)

        print(f"Likelihood: {likelihood}")        
        print(f"Beta prior:     a = {a0}   b = {b0}")
        print(f"Beta posterior: a = {a}  b = {b}")
        print()


def readFile(filename="input.txt"):
    trials = []
    with open(filename, 'r') as file:
        for line in file:
            trials.append(line.strip())

    return trials


def main():
    trials = readFile()
    
    print("Case 1: a = 0, b = 0   ========================================================")
    Case(trials, a=1, b=1)
    print("Case 2: a = 10, b = 1  ========================================================")
    Case(trials, a=10, b=10)


if __name__ == "__main__":
    main()
