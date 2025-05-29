from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv
import networkx as nx
import matplotlib.pyplot as plt
import tqdm

class BinaryCLT:
    """
    Binary Chow-Liu Tree implementation.
    A tree-shaped Bayesian Network learned through the Chow-Liu Algorithm.
    Allows tractable inference for binary variables.
    """

    def __init__(self, data, root: int = None, alpha: float = 0.01):
        """
        Initialize and learn a Chow-Liu Tree from binary data.
        
        Args:
            data: numpy matrix of shape (n_samples, n_features) with values in {0., 1.}
            root: optional root node index (None for random root)
            alpha: smoothing parameter for Laplace correction
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.alpha = alpha
        
        # Learn structure using Chow-Liu algorithm
        self.tree = self._learn_structure(root)
        
        # # Learn parameters (conditional probability tables)
        self.log_params = self.get_log_params()
    
    def _learn_structure(self, root):
        """
        Learn the tree structure using Chow-Liu algorithm.
        The algorithm:
        1. Computes mutual information between all pairs of variables
        2. Finds maximum spanning tree using mutual information as weights
        3. Converts to directed tree with specified root
        
        Returns:
            List of predecessors where tree[i] = j means Xj is parent of Xi.
            Root node has predecessor -1.
        """
        # Compute mutual information matrix between all pairs of variables
        mi_matrix = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                mi = self._compute_mutual_information(i, j)
                mi_matrix[i,j] = mi_matrix[j,i] = mi
        
        # Get maximum spanning tree (using negative MI as weights)
        # Note: minimum_spanning_tree(-M) = maximum_spanning_tree(M)
        mst = minimum_spanning_tree(-mi_matrix)
        
        # Convert sparse MST to dense adjacency matrix
        adj_matrix = mst.toarray()
        # Make it symmetric since MST is undirected
        adj_matrix = adj_matrix + adj_matrix.T
        
        # Convert to predecessor list with specified root
        if root is None:
            root = np.random.randint(0, self.n_features)
            
        # Use breadth-first order to get predecessor list
        _, predecessors = breadth_first_order(adj_matrix, root, directed=True, return_predecessors=True)
        
        # Set root's predecessor to -1
        predecessors[root] = -1

        print(predecessors)
        
        return predecessors
    
    def _compute_mutual_information(self, i, j):
        """
        Compute mutual information between features i and j.
        
        Args:
            i, j: indices of features to compute mutual information between
            
        Returns:
            Mutual information value
        """
        # Count joint occurrences of values
        joint_counts = np.zeros((2, 2))
        for x in self.data:
            joint_counts[int(x[i]), int(x[j])] += 1
            
        # Add Laplace smoothing to avoid zero probabilities
        joint_counts += self.alpha
        
        # Normalize to get joint probabilities
        joint_probs = joint_counts / (self.n_samples + 4*self.alpha)
        
        # Compute marginal probabilities
        p_i = np.sum(joint_probs, axis=1)
        p_j = np.sum(joint_probs, axis=0)
        
        # Compute mutual information: I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
        mi = 0
        for a in range(2):
            for b in range(2):
                if joint_probs[a,b] > 0:
                    mi += joint_probs[a,b] * np.log(joint_probs[a,b] / (p_i[a] * p_j[b]))
        
        return mi
    
    def get_log_params(self):
        """
        Learn the conditional probability tables using maximum likelihood with Laplace correction.
        
        Returns:
            Log of CPTs as a (n_features, 2, 2) array where:
            - First dimension: feature index
            - Second dimension: parent value (0 or 1)
            - Third dimension: feature value (0 or 1)
        """
        log_params = np.zeros((self.n_features, 2, 2))
        
        for i in range(self.n_features):
            parent = self.tree[i]
            
            if parent == -1:  # Root node
                # Count occurrences of each value
                counts = np.zeros(2)
                for x in self.data:
                    counts[int(x[i])] += 1
                
                # Add Laplace smoothing
                counts += 2*self.alpha
                
                # Compute probabilities
                probs = counts / (self.n_samples + 4*self.alpha)
                
                # Store log probabilities (same for both parent values since it's root)
                log_params[i,0,:] = log_params[i,1,:] = np.log(probs)
            else:
                # Count joint occurrences with parent
                joint_counts = np.zeros((2, 2))
                for x in self.data:
                    joint_counts[int(x[parent]), int(x[i])] += 1
                
                # Add Laplace smoothing
                joint_counts += self.alpha
                
                # Compute conditional probabilities
                parent_counts = np.sum(joint_counts, axis=1)
                for a in range(2):
                    if parent_counts[a] > 0:
                        log_params[i,a,:] = np.log(joint_counts[a,:] / parent_counts[a])
        
        return log_params
    
    def get_tree(self):
        """Return the list of predecessors representing the tree structure."""
        return self.tree
    
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
        Simply multiply all conditional probabilities.
        """

        log_prob = 0
        for i in range(self.n_features):
            parent = self.tree[i]
            if parent == -1:  # Root node
                log_prob += self.log_params[i,0,int(x[i])]
            else:
                log_prob += self.log_params[i,int(x[parent]),int(x[i])]
        return log_prob
    
    def _compute_log_prob_marginal_exhaustive(self, x):
        """
        Compute log probability for a marginal query using exhaustive inference.
        Sums over all possible assignments of unobserved variables.
        """
        observed_mask = ~np.isnan(x)
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
    
    def _compute_log_prob_marginal_efficient(self, x):
        """
        Compute log probability for a marginal query using efficient inference.
        Uses variable elimination following the tree structure with the sum-product algorithm.
        
        Args:
            x: numpy array of shape (n_features,) with values in {0., 1., np.nan}
               np.nan indicates unobserved variables
               
        Returns:
            float: log probability of the marginal query
        """
        observed_mask = ~np.isnan(x)
        unobserved_mask = np.isnan(x)
        
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
        leaves = [i for i in range(self.n_features) if i not in self.tree]
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
    
    def sample(self, n_samples: int):
        """
        Generate n_samples i.i.d. samples using ancestral sampling.
        
        Args:
            n_samples: number of samples to generate
        
        Returns:
            numpy array of shape (n_samples, n_features) containing samples
        """

        samples = np.zeros((n_samples, self.n_features), dtype=int)
        
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

    def visualize_tree(self, title="Chow-Liu Tree Structure"):
        """
        Visualize the learned tree structure using networkx and matplotlib.
        Uses a hierarchical layout for a tree-like appearance.
        """
        G = nx.DiGraph()
        for i in range(self.n_features):
            G.add_node(i, label=f'X{i}')
        for i in range(self.n_features):
            if self.tree[i] != -1:
                G.add_edge(self.tree[i], i)

        try:
            # Use graphviz for hierarchical layout
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except ImportError:
            # Fallback to spring layout if graphviz is not available
            pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='lightblue', node_size=1000, arrows=True, arrowsize=20)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show() 