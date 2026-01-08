import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt

def calculate_epsilon_tilde(epsilon, tau, Dk, gamma):
    p = 1 - (1 - 1 / Dk)**tau
    return np.log((np.exp(epsilon)-1)/p + 1)

def calculate_delta(epsilon_tilde, sigma, tau, Dk, gamma, K):
    delta = 0
    for j in range(1, tau + 1):
        term1 = comb(tau, j) * ((1 / Dk)**j) * ((1 - 1 / Dk)**(tau - j)) * ((np.exp(epsilon_tilde) - 1)/(np.exp(epsilon_tilde / j) - 1))
        term2 = norm.cdf((tau * gamma) / (np.sqrt(K) * sigma) - (np.sqrt(K) * epsilon_tilde * sigma) / (2 * j * tau * gamma)) 
        term3 = np.exp(epsilon_tilde / j) * norm.cdf(-(tau * gamma) / (np.sqrt(K) * sigma) - (np.sqrt(K) * epsilon_tilde * sigma) / (2 * j * tau * gamma))
        delta = delta + term1 * (term2 - term3)
    
    return delta

def find_optimal_sigma(epsilon, tau, Dk, gamma, precision_level, K):
    epsilon_tilde = calculate_epsilon_tilde(epsilon, tau, Dk, gamma)
    
    step_size = 10 ** (- precision_level)
    best_sigma = 0
    best_delta = 0


    sigma_range = np.arange(step_size, 0.000001, step_size)
    for sigma in sigma_range:
        delta = calculate_delta(epsilon_tilde, sigma, tau, Dk, gamma, K)
        if delta > best_delta:
            best_delta = delta
            best_sigma = sigma

    return best_sigma, best_delta

# User input and simulation
if __name__ == "__main__":
    epsilon_range = [1.45]
    tau = 15
    Dk = 1666
    dataset_size = Dk
    gamma = 1
    K = 30
    precision_level = 8
    target_delta = 0.01


    epsilon = 1.45
    sigma = 0.001
    epsilon_tilde = calculate_epsilon_tilde(epsilon, tau, dataset_size, gamma)
    delta = calculate_delta(sigma, epsilon_tilde, tau, K, gamma, dataset_size)
    print(f"delta = {delta:.6f}")

    for epsilon in epsilon_range:
        epsilon_tilde = calculate_epsilon_tilde(epsilon, tau, dataset_size, gamma)
        best_delta = 0
        best_sigma = 0
        best_distance = 1
        sigmas = np.arange(0.0000001, 0.0001, 0.0000001)
        for sigma in sigmas:
            delta = calculate_delta(sigma, epsilon_tilde, tau, K, gamma, dataset_size) 

            # deltas = [calculate_delta(sigma, epsilon_tilde, tau, K, gamma, dataset_size) for sigma in sigmas]
            if np.abs(delta - target_delta) < best_distance:
                best_delta = delta
                best_sigma = sigma
                best_distance = np.abs(delta - target_delta)

        print(f"(epsilon, delta, sigma) = ({epsilon:.2f}, {best_delta:.5f}, {best_sigma:.15f})")

        
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, deltas)
    plt.xlabel('sigma')
    plt.ylabel('delta')
    plt.title('Relationship between delta and sigma')
    plt.grid(True)
    plt.yscale('log')  
    plt.show()
    plt.savefig('plot.png')

    """

    """
    for epsilon in epsilon_range:
    # Find optimal sigma
        optimal_sigma, optimal_delta = find_optimal_sigma(epsilon, tau, Dk, gamma, precision_level, K)
        print(f"Optimal (epsilon, delta, sigma): ({epsilon:.6f}, {optimal_delta:.6f}, {optimal_sigma:.10f})")
    """