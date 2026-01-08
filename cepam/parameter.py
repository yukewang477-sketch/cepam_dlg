import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import comb
from scipy.stats import norm
from mpmath import mp

mp.dps = 100  # Set precision to 50 decimal places

def safe_exp(x):
    """
    Safely compute exponential for large values using mpmath
    """
    try:
        if abs(x) > 709:  # numpy's exp limit is around 709
            return float(mp.exp(x))
        return float(np.exp(x))
    except OverflowError:
        return float(mp.exp(x))


def calculate_epsilon_tilde(epsilon: float, tau: int, dataset_size: int) -> float:
    """
    Calculate ε̃ given ε
    
    Returns:
        float: Calculated ε̃ value
    """
    # Calculate p
    p = 1 - (1 - 1/dataset_size)**tau
    epsilon_tilde = float(mp.log(1 + (safe_exp(epsilon) - 1)/p))
    # epsilon_tilde = np.log(1 + (safe_exp(epsilon) - 1)/p)
    print(f"epsilon_tilde: {epsilon_tilde:.4f}")
    return epsilon_tilde

def Phi(x):
    """Standard normal CDF"""
    return norm.cdf(x)

def calculate_delta(sigma: float, epsilon_tilde: float, tau: int, K: int, gamma: float, dataset_size: int) -> float:
    """Calculate δ for given parameters"""
    delta = 0
    for j in range(1, tau + 1):
        coef = comb(tau, j)
        term1 = (1/dataset_size)**j
        term2 = (1 - 1/dataset_size)**(tau-j)
        term3 = (np.exp(epsilon_tilde) - 1)/(np.exp(epsilon_tilde/j) - 1)
        
        phi_term1 = Phi(tau/(np.sqrt(K)*gamma*sigma) - np.sqrt(K)*gamma*epsilon_tilde*sigma/(2*j*tau))
        phi_term2 = np.exp(epsilon_tilde/j) * Phi(-tau/(np.sqrt(K)*gamma*sigma) - np.sqrt(K)*gamma*epsilon_tilde*sigma/(2*j*tau))
        
        delta += coef * term1 * term2 * term3 * abs(phi_term1 - phi_term2)
    return delta
def calculate_laplace_b(epsilon: float, tau: int, K: int, gamma: float, dataset_size: int) -> float:
    """
    计算Laplace机制的b参数
    

    """
    
    epsilon_tilde = float(calculate_epsilon_tilde(epsilon, tau, dataset_size))
    b = 2 * tau / (gamma * epsilon_tilde)
    
    return float(b)
def calculate_epsilon_from_b(b: float, tau: int, gamma: float, dataset_size: int) -> float:
    """
    根据给定的b值反推计算epsilon
    
    Returns:
        float: 计算得到的epsilon值
    """
    # 从 epsilon_tilde >= 2tau/(gamma*b) 得到 epsilon_tilde 的最小值
    epsilon_tilde = 2 * tau / (gamma * b)
    print(epsilon_tilde)
    
    p = 1 - (1 - 1/dataset_size)**tau
    
    epsilon = np.log(p * np.exp(epsilon_tilde))
    
    return epsilon

def plot_delta_sigma_relationship(
        epsilon: float,
        tau: int,
        K: int,
        gamma: float,
        dataset_size: int,
    ):
    """Plot the relationship between delta and sigma"""
    epsilon_tilde = calculate_epsilon_tilde(epsilon, tau, dataset_size)
    
    # 生成一系列sigma值
    sigmas = np.linspace(0.1, 10, 100)
    deltas = [calculate_delta(sigma, epsilon_tilde, tau, K, gamma, dataset_size) for sigma in sigmas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, deltas)
    plt.xlabel('sigma')
    plt.ylabel('delta')
    plt.title('Relationship between delta and sigma')
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度可能更容易看出关系
    plt.show()

# 测试代码
if __name__ == "__main__":
    
    test_params = {
        'epsilon': 700,
        'tau': 15,
        'K': 30,
        'gamma': 1.0, #the scale parameter 3/sqrt(N) where N=message size/lattice dimension
        'dataset_size': 1666, #for MNIST is 50000/30=1666 for CIFAR-10 is 40000/30=1333
        #'dataset_size': 1333,
        #'b': 1
    }
    
    epsilon = 700
    tau = 15
    K = 30
    gamma = 0.1
    dataset_size = 1666
    #print(f"Given b: {test_params['b']}")
    #print(f"Calculated epsilon: {epsilon:.4f}")
    laplace_b = calculate_laplace_b(epsilon, tau, K, gamma, dataset_size)
    print(f"b for Laplace: {laplace_b:.4f}")
    exit()
    #plot_delta_sigma_relationship(**test_params)
    #b = calculate_laplace_b(**test_params)
    #print(f"Calculated b parameter: {b:.4f}")
    """
    sigmas_test = [0.1, 0.5, 1.0, 2.0, 4.0]
    epsilon_tilde = calculate_epsilon_tilde(test_params['epsilon'], 
                                         test_params['tau'], 
                                         test_params['dataset_size'])
    print(f"tau: {test_params['tau']}")
    print(f"K: {test_params['K']}")
    print(f"epsilon: {test_params['epsilon']}")
    print(f"\nEpsilon_tilde: {epsilon_tilde}")
    print("\nDelta values at different sigmas:")
    for sigma in sigmas_test:
        delta = calculate_delta(sigma, epsilon_tilde, 
                              test_params['tau'], 
                              test_params['K'], 
                              test_params['gamma'], 
                              test_params['dataset_size'])
        print(f"sigma = {sigma:.1f}, delta = {delta:.6e}")
    """