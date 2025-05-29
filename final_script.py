from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
import numpy as np
import itertools
import csv

class BinaryCLT:

    def __init__(self, data, root : int = None, alpha : float = 0.01):

        self.data = data
        self.root = root
        self.alpha = alpha

        self.tree = self.get_tree()

        self.log_params = self.get_log_params()

        print(self.tree)

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

                # Applying laplace smoothing
                conditional_frequency += self.alpha

                # Calculate conditional probability
                frequency_of_parent = np.sum(conditional_frequency, axis =1)

                for var in range(2):
                    if frequency_of_parent[var] > 2:
                        log_params[i,var,:] = np.log(conditional_frequency[var,:] / frequency_of_parent[var])

        return log_params

