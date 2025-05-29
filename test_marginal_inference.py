import numpy as np
from binary_clt import BinaryCLT
import matplotlib.pyplot as plt

def create_simple_dataset():
    """
    Create a simple dataset with known probabilities similar to the example in slides.
    """
    # Create a small dataset that would roughly give us the probabilities from slides
    data = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1],
        # Repeat some patterns to get desired probabilities
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    return data

def test_marginal_inference():
    """
    Test and compare exhaustive vs efficient marginal inference.
    """
    # Create dataset and train CLT
    data = create_simple_dataset()
    clt = BinaryCLT(data, root=0, alpha=0.01)
    
    # Visualize the learned tree
    clt.visualize_tree("Test Tree Structure")
    
    # Create test queries
    test_queries = [
        # Query 1: p(x₂=0, x₅=1) from slides
        np.array([np.nan, 0, np.nan, np.nan, 1]),
        
        # Query 2: p(x₂=0, x₃=1, x₄=1) from slides
        np.array([np.nan, 0, 1, 1, np.nan]),
        
        # Query 3: Fully observed case
        np.array([1, 0, 1, 1, 1]),
        
        # Query 4: Only one variable observed
        np.array([np.nan, np.nan, 1, np.nan, np.nan]),
        
        # Query 5: Four variables observed
        np.array([1, 0, 1, 1, np.nan])
    ]
    
    # Compare results
    print("\nComparing exhaustive vs efficient inference:")
    print("-" * 50)
    print(f"{'Query':30} {'Exhaustive':15} {'Efficient':15} {'Diff':10}")
    print("-" * 50)
    
    for i, query in enumerate(test_queries, 1):
        # Get observed variables for query description
        observed = [(f"x{j}={v:.0f}") for j, v in enumerate(query) if not np.isnan(v)]
        query_desc = f"Query {i}: p({', '.join(observed)})"
        
        # Compute both methods
        log_prob_exhaustive = clt.log_prob(query.reshape(1, -1), exhaustive=True)[0]
        log_prob_efficient = clt.log_prob(query.reshape(1, -1), exhaustive=False)[0]
        
        # Print comparison
        print(f"{query_desc:30} {log_prob_exhaustive[0]:15.6f} {log_prob_efficient[0]:15.6f} {abs(log_prob_exhaustive[0] - log_prob_efficient[0]):10.6f}")
    
    print("\nNote: Differences should be very small (close to 0) due to numerical precision.")

def test_random_queries():
    """
    Test random marginal queries to ensure consistency between methods.
    """
    # Create dataset and train CLT
    data = create_simple_dataset()
    clt = BinaryCLT(data, root=0, alpha=0.01)
    
    # Generate random queries
    n_queries = 20
    n_features = data.shape[1]
    
    print("\nTesting random queries:")
    print("-" * 50)
    
    max_diff = 0
    total_diff = 0
    
    for i in range(n_queries):
        # Randomly create observed mask
        observed_mask = np.random.choice([True, False], size=n_features)
        query = np.full(n_features, np.nan)
        
        # Fill observed variables with random 0/1 values
        query[observed_mask] = np.random.choice([0, 1], size=np.sum(observed_mask))
        
        # Compute probabilities
        log_prob_exhaustive = clt.log_prob(query.reshape(1, -1), exhaustive=True)[0]
        log_prob_efficient = clt.log_prob(query.reshape(1, -1), exhaustive=False)[0]
        
        diff = abs(log_prob_exhaustive[0] - log_prob_efficient[0])
        max_diff = max(max_diff, diff)
        total_diff += diff
        
        if i < 5:  # Print first 5 examples
            observed = [(f"x{j}={v:.0f}") for j, v in enumerate(query) if not np.isnan(v)]
            query_desc = f"Random {i+1}: p({', '.join(observed)})"
            print(f"{query_desc:30} {log_prob_exhaustive[0]:15.6f} {log_prob_efficient[0]:15.6f} {diff:10.6f}")
    
    print("-" * 50)
    print(f"Average difference: {total_diff/n_queries:.6f}")
    print(f"Maximum difference: {max_diff:.6f}")

if __name__ == "__main__":
    print("Testing marginal inference with example queries from slides...")
    test_marginal_inference()
    
    print("\nTesting marginal inference with random queries...")
    test_random_queries() 