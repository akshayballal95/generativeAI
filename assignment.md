# 2AMU20 Generative Models Homework Assignment 2

**Release date:** Friday, 9th May 2025  
**Due date:** Thursday, 29th May 2025, 23:59

Solve the tasks below and report your solution in a report in pdf format. Make sure that your report is written in a clean and easily understandable manner. Answer all questions asked and report the results. Please also explain your reasoning and intermediate steps. For every coding task, report the relevant snippets and explain/comment your solution/implementation. Submit your work on Canvas using two files: a pdf report and a single python file with your documented code. Explain your code in the report as often as it is needed to follow your report (the report should be self-contained for the reader). Whenever possible, avoid relying on non-standard external libraries, and document in a way that the code is reproducible. Beware that some tasks may describe extra constraints about your code and about what can or cannot be used - read the instructions carefully. You must solve the assignment with the team to which you have been assigned on Canvas. Please state the team number, as well as the names and student numbers of all students in your pdf report.

## Task 1 - Independence in Graphical Models [20 points]

The image depicts a directed acyclic graph (DAG) with four nodes, labeled *X*₁, *X*₂, *X*₃, and *X*₄.

* *X*₁ has a directed edge pointing to *X*₃.
* *X*₂ has a directed edge pointing to *X*₃.
* *X*₃ has a directed edge pointing to *X*₄.

**Figure 1:** "Extended collider:" A collider X₁ → X₃ ← X₂ with a descendant X₄.

X₁, X₂, X₃ and X₄ are binary, i.e. taking values in {0,1}. Provide CPDs such that in the resulting Bayesian network X₁ ⟂ X₂| X₄. Demonstrate this by showing that the conditional p(X₁, X₂| X₄) does not factorize as p(X₁, X₂|X₄) = p(X₁|X₄) p(X₂ | X₄) for your choice of CPDs.

### Task 1b - CI-based Structure Learning [10 points]

Assume a true distribution p*(X₁, X₂, X₃, X₄) which is given by a Bayesian network with DAG G. While p* is unknown, we have access to an ideal conditional independence (CI) test (oracle CI test), which returns the following results:

| Test | Result | Conditioning Set |
|------|--------|------------------|
| X₁ ⟂ X₂ | True | ∅ |
| X₁ ⟂ X₃ | False | ∅ |
| X₁ ⟂ X₄ | False | ∅ |
| X₂ ⟂ X₃ | True | ∅ |
| X₂ ⟂ X₄ | False | ∅ |
| X₃ ⟂ X₄ | False | ∅ |
| X₁ ⟂ X₂ | True | {X₃} |
| X₁ ⟂ X₄ | True | {X₃} |
| X₂ ⟂ X₄ | True | {X₃} |
| X₁ ⟂ X₂ | False | {X₄} |
| X₁ ⟂ X₃ | True | {X₄} |
| X₂ ⟂ X₃ | True | {X₄} |
| X₃ ⟂ X₄ | False | {X₁, X₂} |
| X₂ ⟂ X₄ | True | {X₁, X₃} |
| X₂ ⟂ X₃ | True | {X₁, X₄} |
| X₁ ⟂ X₄ | True | {X₂, X₃} |
| X₁ ⟂ X₃ | True | {X₂, X₄} |
| X₁ ⟂ X₂ | False | {X₃, X₄} |

**Table 1:** Results of an ideal conditional independence test on p*.

Can the graph G be uniquely determined from the CI tests in Tab. 1? If yes, what is the graph G? If no, determine its skeleton and direct as many edges as possible. Explain your reasoning.

### Task 1c - D-separation and Separation [5 points]

Consider Fig. 2, showing a DAG (left) and an undirected graph (right), representing a Bayesian network structure and Markov network structure over random variables X = {X₁, X₂, X₃, X₄}, respectively. Which conditional independencies of the form X ⟂ Y | Z follow from i) the DAG and ii) the undirected graph? Generate for each graph a table showing their CI statements, akin to Tab. 1.

**Graph 1 (DAG):**
* Nodes: X₁, X₂, X₃, X₄
* Edges: X₁ → X₂, X₁ → X₃, X₂ → X₄, X₃ → X₄

**Graph 2 (Undirected):**
* Nodes: X₁, X₂, X₃, X₄
* Edges: X₁ — X₂, X₁ — X₃, X₂ — X₄, X₃ — X₄

**Figure 2:** A DAG (left) and an undirected graph (right), both over random variables X₁, X₂, X₃, X₄.

## Task 2 - Chow-Liu Trees [80 points]

Implement the python class `BinaryCLT`: A tree-shaped Bayesian Network, learned through the Chow-Liu Algorithm, allowing tractable inference. You are allowed to use the functions `minimum_spanning_tree`, `breadth_first_order`, `logsumexp`, `numpy` and `csv`. The implementation must be organized in a single python file. Make sure to respect the following skeleton:

```python
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv

class BinaryCLT:
    def __init__(self, data, root: int = None, alpha: float = 0.01):
        ...
    
    def get_tree(self):
        ...
    
    def get_log_params(self):
        ...
    
    def log_prob(self, x, exhaustive: bool = False):
        ...
    
    def sample(self, n_samples: int):
        ...
```

**Note:** `minimum_spanning_tree(-M) = maximum_spanning_tree(M)`, where M is a matrix.

The `__init__` method should create and learn a CLT given a binary dataset `data`, i.e. a numpy matrix whose rows correspond to samples, whose columns correspond to random variables and whose entries are in {0., 1.}, an optional root node `root` (None by default) and a smoothing parameter `alpha` (0.01 by default).

```python
def __init__(self, data, root=None, alpha=0.01):
    # your code
```

The input `root` must be `None` or a non-negative integer in range `(data.shape[1])`. For instance, if `root == 0` the tree must be rooted at X₀. If `root == None` the tree shall be rooted at a random node. The smoothing parameter `alpha` is a non-negative real number, by default `alpha = 0.01`. Estimate the probabilities in the CPTs via maximum likelihood estimation and using Laplace's correction, i.e.:

```
p(Y = y|Z = z) = p(Y = y, Z = z) / p(Z = z)

p(Y = y, Z = z) = (α + Σₓ∈D 1[x[Y] = y, x[Z] = z]) / (4α + |D|)

p(Z = z) = (2α + Σₓ∈D 1[x[Z] = z]) / (4α + |D|)
```

Remember to work, when possible, in the logarithmic domain for numerical stability: products become sums and sums become log-sum-exp operations.² Use the natural logarithm.

You can download binary datasets for density estimation via this link: every folder corresponds to a dataset which is splitted in training, validation and test. You can easily load the dataset with `csv.reader`:

```python
with open('file_name.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    dataset = np.array(list(reader)).astype(np.float)
```

Use these datasets to test your code. You are not required to report results for all of them, but results for the nltcs dataset are required on Task 2e.

### Task 2a - Structure Learning [20 points]

The `get_tree` method should return the list of predecessors of the learned structure: If Xⱼ is the parent of Xᵢ then `tree[i] = j`, while, if Xᵢ is the root of the tree then `tree[i] = -1`. For instance:

```python
def get_tree(self):
    # your code
    return tree

# for the CLT in Fig. 3: tree = [-1, 0, 4, 4, 0]
```

**Figure 3:** Structure of a learned CLT.
* Root node: X₀
* X₀ has children: X₄ and X₁
* X₄ has children: X₃ and X₂
* Leaf nodes: X₁, X₂, X₃

### Task 2b - Parameter Learning [15 points]

The `get_log_params` method should return the log of the conditional probability tables (CPTs) of your CLT, estimated by maximum likelihood and using Laplace's correction. Formally, the method returns `log_params` a D × 2 × 2 - dimensional array such that `log_params[i, j, k] = log p(xᵢ = k|x_{π(i)} = j)`, where d is the number of RVs and π(i) is the index of Xᵢ's parent. For simplicity, when Xᵢ is the root of the tree then `log_params[i, 0, k] = log_params[i, 1, k]` for every k. For instance:

```python
def get_log_params(self):
    # your code
    return log_params

# for the tree on the right, return:
# log_params = array([
#     [[-1.204, -0.357], [-1.204, -0.357]],  # X₀
#     [[-1.609, -0.223], [-0.511, -0.916]],  # X₁|X₀
#     [[-0.916, -0.511], [-2.303, -0.105]],  # X₂|X₄
#     [[-0.223, -1.609], [-0.693, -0.693]],  # X₃|X₄
#     [[-0.105, -2.303], [-0.916, -0.511]]   # X₄|X₀
# ])
```

**Figure 4:** A CLT and its CPTs.

Example CPTs:
- p(X₀): [3/5, 2/5]
- p(X₄|X₀): [[9/10, 1/10], [1/6, 5/6]]
- p(X₁|X₀): [[2/5, 3/5], [4/10, 6/10]]
- p(X₃|X₄): [[8/10, 2/5], [2/5, 3/5]]
- p(X₂|X₄): [[4/10, 1/10], [6/10, 9/10]]

Note that the `log_params` above are rounded to 3 decimal places just to avoid clutter and that `numpy.exp(log_params)` creates the CPTs in Figure 4.

### Task 2c - Inference [25 points]

The `log_prob` method should compute the log probability for both fully observed samples and marginal queries. Formally, the method takes a N × D - dimensional float array `x` as input, representing a set of samples - for marginal queries, some values are marked as missing (encoded as nans).

In particular, every row in `x` refers to a query and the ith column to RV Xᵢ. If Xᵢ is observed in a query then it takes either 0. or 1. as value. Otherwise, if Xᵢ is missing its value is set to `numpy.nan` ('not a number'). Let, for a specific query, Y be the set of observed RVs and Z be the set of unobserved RVs. For the query variables, let y be the observed values in the corresponding row in x. The method `log_prob` should return:

```
log p(y) = log Σ_z p(y, z)
```

for each row in x. Of course, when Y = X, the result is simply the log-probability for a fully observed sample.

The method should return a N × 1 - dimensional float array `lp` such that `lp[i]` contains the log probability of `x[i]`. For instance:

```python
def log_prob(self, x, exhaustive=False):
    # your code
    return lp

# Example:
x = np.array([
    [1., 0., 1., 1., 0.],
    [np.nan, 0., np.nan, np.nan, 1.],
    [np.nan, 0., 1., 1., np.nan]
])
lp = np.array([[-3.904], [-1.354], [-1.946]])
```

Therefore, the first row of x (i.e. `x[0]`) corresponds to the query `log p(X₀ = 1, X₁ = 0, X₂ = 1, X₃ = 1, X₄ = 0)`, while the second (i.e. `x[1]`) corresponds to `log p(X₁ = 0, X₄ = 1)`. Note that `lp` is rounded to 3 decimal places just to avoid clutter.

Implement both exhaustive inference and efficient inference: If the `exhaustive` flag is `True`, perform exhaustive inference, i.e., compute the whole joint distribution and perform inference by explicitly summing out the unobserved variables Z from the joint. If the `exhaustive` flag is `False`, perform an efficient inference algorithm, such as variable elimination (see lectures).

**Hint:** A sanity check: Assuming that your CPTs (i.e. `log_params`) have been correctly estimated and that your `log_prob` method is correctly implemented, the sum of the probabilities (i.e. `numpy.sum(numpy.exp(lp))`) of all 2^D possible fully-observed states must be 1 (up to numerical artifacts).

### Task 2d - Ancestral Sampling [10 points]

The `sample` method should produce i.i.d. samples from the CLT distribution using ancestral sampling. Formally, the method takes an integer `n_samples` as input and outputs an `n_samples × D` - dimensional array of integers containing a sample for every row, where D is the number of RVs. For instance:

```python
def sample(self, n_samples):
    # your code
    return samples

# Example (output is random, of course):
n_samples = 3
samples = np.array([
    [1, 0, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0]
])
```

Therefore, the first row of samples (i.e. `samples[0]`) corresponds to the sample (x₀ = 1, x₁ = 0, x₂ = 1, x₃ = 1, x₄ = 0).

### Task 2e - NLTCS Data [10 points]

Learn a CLT over the nltcs training set (which you can find under this link) and set `root = 0` and `alpha = 0.01`. Do not use the validation or test set for learning. Report:

* The tree (i.e. the list of predecessors and a plot of it)
* The CPTs (i.e. `log_params`)
* The average train and average test log-likelihoods, i.e., the average over sample-wise log-likelihoods, averaged over the train and test sets, respectively

Moreover, compute the log probabilities over the marginal queries found in this file. Perform the inference with both `exhaustive=True` and `exhaustive=False`. Do they deliver the same results for `lp`?

* Take a coarse runtime measurement³ when running inference with `exhaustive=True` and `exhaustive=False` for the marginal queries in the previous point. Report the runtime difference and explain where it comes from. What would happen if you run a marginal query with `exhaustive=True` on a dataset like accidents?

* Use the `sample` method to produce 1000 samples from the CLT distribution. Evaluate and report the average log-likelihood over these 1000 samples. Is it in a similar range as the average test log-likelihood?

---

**Footnotes:**
² See the function `logsumexp` from `scipy.special`.
³ For example, using `start_t = time.time(); ... your code ...; elapsed_time = time.time() - start_t`.