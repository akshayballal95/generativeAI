import time
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
import numpy as np
import itertools
import csv

import tqdm

class BinaryCLT:

    def __init__(self, data, root : int = None, alpha : float = 0.01):

        self.data = data
        self.root = root
        self.alpha = alpha

        self.tree = self.get_tree()

        self.log_params = self.get_log_params()

    def get_tree(self):

        def mutual_information_calculation(data, alpha,  i , j):
            contingency_matrix = np.zeros((2,2))
            for individual_row in data:
                contingency_matrix[int(individual_row[i]), int(individual_row[j])] += 1

            contingency_matrix += alpha

            joint_probability = contingency_matrix / (data.shape[0] + 4* alpha)

            marginal_i = np.sum(joint_probability, axis = 1)
            marginal_j = np.sum(joint_probability, axis = 0)

            mutual_information = 0
            for a in range(joint_probability.shape[0]):
                for b in range(joint_probability.shape[1]):
                    mutual_information += joint_probability[a,b] * np.log(joint_probability[a,b] / (marginal_i[a] * marginal_j[b]))

            return mutual_information
        
        mutual_information_matrix = np.zeros((self.data.shape[1], self.data.shape[1]))

        for i in range(self.data.shape[1]):
            for j in range(i+1 , self.data.shape[1]):
                mutual_information = mutual_information_calculation(self.data , self.alpha, i, j)
                mutual_information_matrix[i,j] = mutual_information
                mutual_information_matrix[j,i] = mutual_information

        minimum_tree = minimum_spanning_tree(-mutual_information_matrix)

        adjacency_matrix = minimum_tree.toarray()

        adjacency_matrix = adjacency_matrix + adjacency_matrix.T

        if self.root is None:
            self.root = np.random.randint(0, self.data.shape[1])

        _, predecessors = breadth_first_order(adjacency_matrix, self.root, directed=True, return_predecessors=True)

        predecessors[self.root] = -1

        return predecessors
    
    def get_log_params(self):

        log_params = np.zeros((self.data.shape[1], 2, 2))

        for i in range(self.data.shape[1]):

            parent = self.tree[i]

            if parent == -1:
                # Counting two outcomes 0 and 1
                frequency = np.zeros(2)

                for j in self.data:
                    frequency[int(j[i])] += 1
                
                # Applying laplace smoothing
                frequency += 2*self.alpha

                # Calculating prior probability
                prior_prob = frequency / (self.data.shape[0] + 4*self.alpha)

                # Assigning log params
                log_params[i,0,:] = log_params[i,1,:] = np.log(prior_prob) # Will be the same since there is no conditional dependence
        
            else:

                conditional_frequency = np.zeros((2,2))

                for j in self.data:
                    conditional_frequency[int(j[parent]), int(j[i])] += 1


                # Calculate conditional probability
                frequency_of_parent = np.sum(conditional_frequency, axis =1)

                for var in range(2):
                    log_params[i,var,:] = np.log( (conditional_frequency[var,:] + 2*self.alpha) / (frequency_of_parent[var] + 4*self.alpha))

        return log_params
    
    
    def log_prob(self, x, exhaustive: bool = False):
        """
        Compute log probability for fully observed samples or marginal queries.
    
        """
        n_queries = x.shape[0]
        lp = np.zeros((n_queries, 1))
        
        if exhaustive:
            # Exhaustive inference: sum over all possible assignments of unobserved variables
            for i in tqdm.tqdm(range(n_queries), desc="Exhaustive inference"):
                query = x[i]
                observed_mask = ~np.isnan(query)
                
                if np.all(observed_mask):  # Fully observed
                    lp[i] = self._compute_log_prob_fully_observed(query)
                else:  # Marginal query
                    lp[i] = self._compute_log_prob_marginal_exhaustive(query)
        else:
            # Efficient inference using variable elimination
            for i in range(n_queries):
                query = x[i]
                observed_mask = ~np.isnan(query)
                
                if np.all(observed_mask):  # Fully observed
                    lp[i] = self._compute_log_prob_fully_observed(query)
                else:  # Marginal query
                    lp[i] = self._compute_log_prob_marginal_efficient(query)
        
        return lp
    
    def sample(self, n_samples: int):
        """
        Generate n_samples i.i.d. samples using ancestral sampling.
        
        Args:
            n_samples: number of samples to generate
        
        Returns:
            numpy array of shape (n_samples, n_features) containing samples
        """

        samples = np.zeros((n_samples, self.data.shape[1]), dtype=int)
        
        # Get sampling order (top-down in tree)
        sampling_order = []
        visited = set()
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            sampling_order.append(node)
            children = np.where(self.tree == node)[0]
            for child in children:
                visit(child)
        
        # Start from root
        root = np.where(self.tree == -1)[0][0]
        visit(root)
        
        # Generate samples
        for i in range(n_samples):
            for node in sampling_order:
                parent = self.tree[node]
                if parent == -1:  # Root node
                    probs = np.exp(self.log_params[node,0,:])
                else:
                    probs = np.exp(self.log_params[node,samples[i,parent],:])
                samples[i,node] = np.random.choice(2, p=probs)
        
        return samples 


    def _compute_log_prob_fully_observed(self, x):
        """
        Compute log probability for a fully observed sample.
        """
        log_prob = 0
        for i in range(self.data.shape[1]):
            parent = self.tree[i]
            if parent == -1:  # Root node
                log_prob += self.log_params[i,0,int(x[i])]
            else:
                log_prob += self.log_params[i,int(x[parent]),int(x[i])]
        return log_prob
    
    def _compute_log_prob_marginal_exhaustive(self, x):
        """
        Compute log probability for a marginal query using exhaustive inference.
        """
        unobserved_mask = np.isnan(x)
        n_unobserved = np.sum(unobserved_mask)
        
        # Generate all possible assignments for unobserved variables
        unobserved_indices = np.where(unobserved_mask)[0]
        log_prob = -np.inf
        
    
        for combinations in itertools.product([0,1], repeat=n_unobserved):
            # Create complete assignment
            complete_x = x.copy()
            complete_x[unobserved_indices] = combinations
            
            # Compute log probability
            log_prob = np.logaddexp(log_prob, self._compute_log_prob_fully_observed(complete_x))
        
        return log_prob
    
    def _compute_log_prob_marginal_efficient(self, x):
        """
        Compute log probability for a marginal query using efficient inference with variable elimination.
        """
        observed_mask = ~np.isnan(x)
        
        # Get elimination order (bottom-up in tree)
        elimination_order = []
        visited = set()
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            children = np.where(self.tree == node)[0]
            for child in children:
                visit(child)
            elimination_order.append(node)
        
        # Start from leaves
        leaves = [i for i in range(self.data.shape[1]) if i not in self.tree]
        for leaf in leaves:
            visit(leaf)
            
        # Visit remaining nodes to complete elimination order
        root = np.where(self.tree == -1)[0][0]
        visit(root)
        
        # Initialize messages dictionary
        # messages[i][parent_val] = message from node i to its parent when parent has value parent_val
        messages = {}
        
        # Compute messages from leaves to root
        for node in elimination_order:
            if node == root:
                continue
                
            parent = self.tree[node]
            messages[node] = {}
            
            # Get children's messages to this node
            children = np.where(self.tree == node)[0]
            
            # If node is observed
            if observed_mask[node]:
                node_val = int(x[node])
                # Message when parent = 0
                msg_0 = self.log_params[node, 0, node_val]
                # Message when parent = 1
                msg_1 = self.log_params[node, 1, node_val]
                
                # Add children's messages if any
                for child in children:
                    msg_0 += messages[child][0]
                    msg_1 += messages[child][1]
                    
                messages[node][0] = msg_0
                messages[node][1] = msg_1
                
            # If node is unobserved
            else:
                # For each parent value
                for parent_val in [0, 1]:
                    # Sum over node values
                    msg = -np.inf  # Initialize in log space
                    for node_val in [0, 1]:
                        curr_msg = self.log_params[node, parent_val, node_val]
                        
                        # Add children's messages
                        for child in children:
                            curr_msg += messages[child][node_val]
                            
                        msg = np.logaddexp(msg, curr_msg)
                    
                    messages[node][parent_val] = msg
        
        # Compute final probability at root
        root_children = np.where(self.tree == root)[0]
        
        if observed_mask[root]:
            root_val = int(x[root])
            log_prob = self.log_params[root, 0, root_val]  # Root's prior
            
            # Add children's messages
            for child in root_children:
                log_prob += messages[child][root_val]
                
        else:
            # Sum over root values
            log_prob = -np.inf  # Initialize in log space
            for root_val in [0, 1]:
                curr_prob = self.log_params[root, 0, root_val]  # Root's prior
                
                # Add children's messages
                for child in root_children:
                    curr_prob += messages[child][root_val]
                    
                log_prob = np.logaddexp(log_prob, curr_prob)
            
        return log_prob
    

### Testing on NLTCS Dataset

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


train_data = load_data('datasets/nltcs/nltcs.train.data')
test_data = load_data('datasets/nltcs/nltcs.test.data')
marginal_data = load_data('nltcs_marginals.data')


# Create and train CLT
print("\nTraining CLT...")
clt = BinaryCLT(train_data, root=0, alpha=0.01)

# Get and display tree structure (Task 2e.1)
tree = clt.get_tree()
print("\nTree structure (predecessors):")
tree_copy = tree.copy()
tree_copy[1:] = tree_copy[1:] + 1

# CPTs Task 2e.2
print(clt.log_params.transpose((0, 2,1)))

# Compute and display average log-likelihoods for fully observed data (Task 2e.3)
print("\nComputing log-likelihoods...")
train_ll = np.mean(clt.log_prob(train_data))
test_ll = np.mean(clt.log_prob(test_data))
print(f"Average train log-likelihood: {train_ll:.4f}")
print(f"Average test log-likelihood: {test_ll:.4f}")


# Sanity check: verify that probabilities sum to 1
print("\nPerforming sanity check...")
n_features = train_data.shape[1]
# Generate all possible states
all_states = np.array(list(itertools.product([0, 1], repeat=n_features)))
# Compute probabilities
all_probs = np.exp(clt.log_prob(all_states))
total_prob = np.sum(all_probs)
print(f"Sum of probabilities of all possible states: {total_prob:.6f}")


# Compute and display average log-likelihoods for marginal data 

 # Compare inference methods (Task 2e.5)
print("\nComparing inference methods...")

# Test exhaustive inference
start_t = time.time()
print("Shape of test_data: ", test_data.shape)
lp_exhaustive = clt.log_prob(marginal_data, exhaustive=True)
exhaustive_time = time.time() - start_t


# Display results
print("\nResults for test queries:")
print("Exhaustive inference:")
# print(lp_exhaustive)
print(np.mean(lp_exhaustive))
print(f"Time taken: {exhaustive_time:.4f} seconds")

# Test efficient inference
start_t = time.time()
lp_efficient = clt.log_prob(marginal_data, exhaustive=False)
efficient_time = time.time() - start_t


print("\nEfficient inference:")
# print(lp_efficient)
print(np.mean(lp_efficient))
print(f"Time taken: {efficient_time:.4f} seconds")

# Compare results between methods
print("\nDifference in results:")
print(np.mean(np.abs(lp_exhaustive - lp_efficient)))
print("Efficient inference is faster than exhaustive inference by", (exhaustive_time - efficient_time)/exhaustive_time*100, "%")




# Test sampling (Task 2e.6)
print("\nGenerating and evaluating samples...")
n_samples = 1000
samples = clt.sample(n_samples)
sample_ll = np.mean(clt.log_prob(samples))
print(f"Average log-likelihood of {n_samples} samples: {sample_ll:.4f}")
