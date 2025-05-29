import numpy as np
from binary_clt import BinaryCLT

def create_test_dataset():
    """Create a simple dataset for testing"""
    data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    return data

def test_marginal_vs_joint():
    """Compare marginal and joint log likelihoods"""
    # Create and train CLT
    data = create_test_dataset()
    clt = BinaryCLT(data, root=0, alpha=0.01)
    
    # Create all possible fully observed states
    all_states = np.array(list(itertools.product([0, 1], repeat=3)))
    
    # Create marginal queries with different numbers of observed variables
    queries = {
        "1 observed": np.array([[0, np.nan, np.nan],
                              [1, np.nan, np.nan]]),
        "2 observed": np.array([[0, 0, np.nan],
                              [0, 1, np.nan],
                              [1, 0, np.nan],
                              [1, 1, np.nan]]),
        "fully observed": all_states
    }
    
    print("Comparing average log likelihoods:")
    print("-" * 50)
    print(f"{'Query type':20} {'Avg Log Likelihood':20}")
    print("-" * 50)
    
    for query_type, query in queries.items():
        log_probs = clt.log_prob(query)
        avg_ll = np.mean(log_probs)
        print(f"{query_type:20} {avg_ll:20.6f}")
    
    # Verify that probabilities sum to 1 for each marginalization
    print("\nVerifying probability sums:")
    print("-" * 50)
    print(f"{'Query type':20} {'Sum of probabilities':20}")
    print("-" * 50)
    
    for query_type, query in queries.items():
        log_probs = clt.log_prob(query)
        prob_sum = np.sum(np.exp(log_probs))
        print(f"{query_type:20} {prob_sum:20.6f}")

if __name__ == "__main__":
    import itertools
    test_marginal_vs_joint() 