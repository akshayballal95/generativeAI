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

        #Calculation of mutual information matrix
        def mutual_information_calculation(data, alpha,  i , j):
            contingency_matrix = np.zeros((2,2))

            #Getting the counts of the outcomes
            for individual_row in data:
                contingency_matrix[int(individual_row[i]), int(individual_row[j])] += 1

            #Applying smoothening
            contingency_matrix += alpha

            # Joint probability between variables i and j
            joint_probability = contingency_matrix / (data.shape[0] + 4* alpha)

            # Marginal probability of both a and b
            marginal_i = np.sum(joint_probability, axis = 1)
            marginal_j = np.sum(joint_probability, axis = 0)

            mutual_information = 0
            # Calculation Mutual information value
            for a in range(joint_probability.shape[0]):
                for b in range(joint_probability.shape[1]):
                    mutual_information += joint_probability[a,b] * np.log(joint_probability[a,b] / (marginal_i[a] * marginal_j[b]))

            return mutual_information
        
        mutual_information_matrix = np.zeros((self.data.shape[1], self.data.shape[1]))

        # Calculating the MI matrix between all the variables
        for i in range(self.data.shape[1]):
            for j in range(i+1 , self.data.shape[1]):
                mutual_information = mutual_information_calculation(self.data , self.alpha, i, j)
                mutual_information_matrix[i,j] = mutual_information
                mutual_information_matrix[j,i] = mutual_information

        # Forming a tree based on the maximum MI value between variables
        minimum_tree = minimum_spanning_tree(-mutual_information_matrix)

        adjacency_matrix = minimum_tree.toarray()

        # Making it undirected between the nodes since minimum spanning tree returns directed connections
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T

        if self.root is None:
            self.root = np.random.randint(0, self.data.shape[1])

        _, predecessors = breadth_first_order(adjacency_matrix, self.root, directed=True, return_predecessors=True)

        # Setting the parent of root node as -1
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

                # Applying laplace smoothing
                conditional_frequency += self.alpha

                # Calculate conditional probability
                frequency_of_parent = np.sum(conditional_frequency, axis =1)

                for var in range(2):
                    log_params[i,var,:] = np.log(conditional_frequency[var,:] / frequency_of_parent[var])

        return log_params
    
    
    def log_prob(self, x, exhaustive: bool = False):
        """
        Compute log probability for fully observed samples or marginal queries.
        
        Args:
            x: numpy array of shape (n_queries, n_features) with values in {0., 1., np.nan}
               np.nan indicates missing values for marginal queries
            exhaustive: if True, use exhaustive inference; if False, use efficient inference
        
        Returns:
            numpy array of shape (n_queries, 1) containing log probabilities
        """
        n_queries = x.shape[0]
        lp = np.zeros((n_queries, 1))
        
        if exhaustive:
            # Exhaustive inference: sum over all possible assignments of unobserved variables
            for i in tqdm.tqdm(range(n_queries), desc="Exhaustive inference"):
                query = x[i]
                observed_mask = ~np.isnan(query)
                unobserved_mask = np.isnan(query)
                
                if np.all(observed_mask):  # Fully observed
                    lp[i] = self._compute_log_prob_fully_observed(query)
                else:  # Marginal query
                    lp[i] = self._compute_log_prob_marginal_exhaustive(query)
        else:
            # Efficient inference using variable elimination
            for i in range(n_queries):
                query = x[i]
                observed_mask = ~np.isnan(query)
                unobserved_mask = np.isnan(query)
                
                if np.all(observed_mask):  # Fully observed
                    lp[i] = self._compute_log_prob_fully_observed(query)
                else:  # Marginal query
                    lp[i] = self._compute_log_prob_marginal_efficient(query)
        
        return lp
    

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
        
    
        for assignment in itertools.product([0,1], repeat=n_unobserved):
            # Create complete assignment
            complete_x = x.copy()
            complete_x[unobserved_indices] = assignment
            
            # Compute log probability
            log_prob = np.logaddexp(log_prob, self._compute_log_prob_fully_observed(complete_x))
        
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

# Get and display tree structure
tree = clt.get_tree()
print("\nTree structure (predecessors):")
tree_copy = tree.copy()
tree_copy[1:] = tree_copy[1:] + 1

# Compute and display average log-likelihoods for fully observed data
print("\nComputing log-likelihoods...")
train_ll = np.mean(clt.log_prob(train_data))
test_ll = np.mean(clt.log_prob(test_data))
print(f"Average train log-likelihood: {train_ll:.4f}")
print(f"Average test log-likelihood: {test_ll:.4f}")

print(clt.log_params.transpose((0, 2,1)))


