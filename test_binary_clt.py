import numpy as np
import csv
import time
import itertools
from binary_clt import BinaryCLT

def load_data(filename):
    """
    Load data from a CSV file.
    
    Args:
        filename: path to the CSV file containing binary data
        
    Returns:
        numpy array of shape (n_samples, n_features) with values in {0., 1.}
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        return np.array(list(reader)).astype(np.float64)

def main():
    """
    Main test function that evaluates the BinaryCLT implementation.
    Tests structure learning, parameter learning, inference, and sampling.
    """
    # Load NLTCS dataset
    print("Loading NLTCS dataset...")
    try:
        train_data = load_data('datasets/nltcs/nltcs.train.data')
        test_data = load_data('datasets/nltcs/nltcs.test.data')
        marginal_data = load_data('nltcs_marginals.data')
        print(f"Loaded data shapes - Train: {train_data.shape}, Test: {test_data.shape}")
    except FileNotFoundError:
        print("Error: Could not find NLTCS dataset files.")
        print("Please ensure the files are in the correct location: datasets/nltcs/")
        return
    
    # Create and train CLT
    print("\nTraining CLT...")
    clt = BinaryCLT(train_data, root=0, alpha=0.01)
    
    # Get and display tree structure
    tree = clt.get_tree()
    print("\nTree structure (predecessors):")
    print(tree)
    
    # Get and display parameters
    log_params = clt.get_log_params()
    print("\nLog parameters shape:", log_params.shape)
    
    # Compute and display average log-likelihoods
    print("\nComputing log-likelihoods...")
    train_ll = np.mean(clt.log_prob(train_data))
    test_ll = np.mean(clt.log_prob(test_data))
    print(f"Average train log-likelihood: {train_ll:.4f}")
    print(f"Average test log-likelihood: {test_ll:.4f}")
    
    # Test marginal queries
    print("\nTesting marginal queries...")
    # Create example queries:
    # 1. Query p(X₁=0, X₄=1) - only X₁ and X₄ are observed
    # 2. Query p(X₁=0, X₂=1, X₃=1) - only X₁, X₂, and X₃ are observed
    # 3. Fully observed query - all variables are observed
    # marginal_queries = np.array([
    #     [np.nan, 0., np.nan, np.nan, 1.],  # Query p(X₁=0, X₄=1)
    #     [np.nan, 0., 1., 1., np.nan],      # Query p(X₁=0, X₂=1, X₃=1)
    #     [1., 0., 1., 1., 0.]               # Fully observed query
    # ])
    
    # Compare inference methods
    print("\nComparing inference methods...")
    
    # Test exhaustive inference
    start_t = time.time()
    print("Shape of test_data: ", test_data.shape)
    lp_exhaustive = clt.log_prob(marginal_data[:1000], exhaustive=True)
    exhaustive_time = time.time() - start_t

    
    # Display results
    print("\nResults for test queries:")
    print("Exhaustive inference:")
    # print(lp_exhaustive)
    print(np.mean(lp_exhaustive))
    print(f"Time taken: {exhaustive_time:.4f} seconds")
    
    # Test efficient inference
    start_t = time.time()
    lp_efficient = clt.log_prob(marginal_data[:1000], exhaustive=False)
    efficient_time = time.time() - start_t

    
    print("\nEfficient inference:")
    # print(lp_efficient)
    print(np.mean(lp_efficient))
    print(f"Time taken: {efficient_time:.4f} seconds")
    
    # Compare results between methods
    print("\nDifference in results:")
    print(np.mean(np.abs(lp_exhaustive - lp_efficient)))


    # generate queries with all 2^D possible states for D = 16

   

    # compute the log-likelihood of the queries
    
    # Test sampling
    print("\nGenerating and evaluating samples...")
    n_samples = 10000
    samples = clt.sample(n_samples)
    sample_ll = np.mean(clt.log_prob(samples))
    print(f"Average log-likelihood of {n_samples} samples: {sample_ll:.4f}")
    
    # Sanity check: verify that probabilities sum to 1
    print("\nPerforming sanity check...")
    n_features = train_data.shape[1]
    # Generate all possible states
    all_states = np.array(list(itertools.product([0, 1], repeat=n_features)))
    # Compute probabilities
    all_probs = np.exp(clt.log_prob(all_states))
    total_prob = np.sum(all_probs)
    print(f"Sum of probabilities of all possible states: {total_prob:.6f}")


if __name__ == "__main__":
    main() 