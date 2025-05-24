## Chow-Liu Trees

## Generative AI Models - Lecture 5

9 th May 2025

Thomas Krak (slides adapted from Gennaro Gala) Uncertainty in Artificial Intelligence

<!-- image -->

## Idea: structure learning as discrete optimization

- · Let X be a set of RVs and D = { x n } N n =1 be i.i.d. data
- · Let [ G ] be some family of DAGs over X
- · Define a suitable score S G D ( , )
- · Find G ∗ = arg max G∈ G [ ] S G D ( , )

- · [ G ] is the set of all directed trees [ T ] over X
- · Directed tree : Every RV has at most one parent
- · Score S G D ( , ) = max Θ L G ( , Θ , D ), where
- · Θ are all BN parameters for G (for categorical CPDs)
- · L G ( , Θ , D ) is the log-likelihood
- · Thus, tree G is 'better' than G ' if the log-likelihood of G is higher than the log-likelihood of G ' (when equipping them with their ML parameters):

<!-- image -->

- · Remarkable: Poly-time Algorithm!

## Algorithm 3 VE\_PRI(N, Q, J)

## input:

N:

Q

- variables in network N

Bayesian network

- J: ordering of network variables not in Q

output: the marginal Pr(Q) prior

## main:

- 1:
- 3:
- 4:
- 5: replace all factors fk in $ by factor fi
- 6: end for
- 7: return IIfes f
- = 1 to length of order I do

## Approximating Discrece Probability Distributions with Dependence Trees

AND C. N. LIU, MEMBER, IEEE

<!-- image -->

## Learning Chow-Liu Trees

Inference in Tree-shaped BNs

Ancestral Sampling

## Learning Chow-Liu Trees

(Kullback-Leibler divergence) Let p and q be probability distributions over the same state space X . The Kullback-Leibler divergence between p and q is defined as:

<!-- formula-not-decoded -->

(Kullback-Leibler divergence) Let p and q be probability distributions over the same state space X . The Kullback-Leibler divergence between p and q is defined as:

<!-- formula-not-decoded -->

In other words, it is the expectation of the logarithmic difference between the distributions p and q , where the expectation is taken using the distribution p , i.e.:

<!-- formula-not-decoded -->

(Kullback-Leibler divergence) Let p and q be probability distributions over the same state space X . The Kullback-Leibler divergence between p and q is defined as:

<!-- formula-not-decoded -->

In other words, it is the expectation of the logarithmic difference between the distributions p and q , where the expectation is taken using the distribution p , i.e.:

<!-- formula-not-decoded -->

̸

Note that, in general, KL ( p || q ) = KL ( q || p ) and that KL ( p || q ) = 0 iff p = q .

(Mutual Information) Given two jointly discrete RVs X and Y with joint distribution p XY and marginal distributions p X and p Y , the mutual information MI( X Y ; ) between X and Y is:

<!-- formula-not-decoded -->

(Mutual Information) Given two jointly discrete RVs X and Y with joint distribution p XY and marginal distributions p X and p Y , the mutual information MI( X Y ; ) between X and Y is:

<!-- formula-not-decoded -->

- · The mutual information of two RVs is a measure of the mutual dependence

(Mutual Information) Given two jointly discrete RVs X and Y with joint distribution p XY and marginal distributions p X and p Y , the mutual information MI( X Y ; ) between X and Y is:

<!-- formula-not-decoded -->

- · The mutual information of two RVs is a measure of the mutual dependence
- · Note that, MI is measured in nats (natural unit of information) when the natural logarithm is used.

| p XY ( X Y , )   | x = 0   | x = 1   | p Y ( Y )   |
|------------------|---------|---------|-------------|
| y = 0            | 0 1 .   | 0 3 .   | 0 4 .       |
| y = 1            | 0 2 .   | 0 4 .   | 0 6 .       |
| p X ( X )        | 0 3 .   | 0 7 .   |             |

| p XY ( X Y , )   | x = 0   | x = 1   | p Y ( Y )   |
|------------------|---------|---------|-------------|
| y = 0            | 0 1 .   | 0 3 .   | 0 4 .       |
| y = 1            | 0 2 .   | 0 4 .   | 0 6 .       |
| p X ( X )        | 0 3 .   | 0 7 .   |             |

<!-- formula-not-decoded -->

| p XY ( X Y , )   | x = 0   | x = 1   | p Y ( Y )   |
|------------------|---------|---------|-------------|
| y = 0            | 0 08 .  | 0 32 .  | 0 4 .       |
| y = 1            | 0 12 .  | 0 48 .  | 0 6 .       |
| p X ( X )        | 0 2 .   | 0 8 .   |             |

| p XY ( X Y , )   | x = 0   | x = 1   | p Y ( Y )   |
|------------------|---------|---------|-------------|
| y = 0            | 0 08 .  | 0 32 .  | 0 4 .       |
| y = 1            | 0 12 .  | 0 48 .  | 0 6 .       |
| p X ( X )        | 0 2 .   | 0 8 .   |             |

<!-- formula-not-decoded -->

## Bayesian Networks

A Bayesian Network (BN) over RVs X = ( X i ) d i =1 is a pair ( G P , ), where:

- · G is a DAG which has RVs X as nodes;
- · P is a collection of distributions p X ( i | pa ( X i ));

## and where:

<!-- formula-not-decoded -->

## Bayesian Networks

A Bayesian Network (BN) over RVs X = ( X i ) d i =1 is a pair ( G P , ), where:

- · G is a DAG which has RVs X as nodes;
- P is a collection of distributions p X ( i | pa ( X i ));

<!-- formula-not-decoded -->

<!-- image -->

p ( X ) = p X ( 1 | X 2 , X 5 ) p X ( 2 ) p X ( 3 | X 1 ) p X ( 4 | X 1 ) p X ( 5 )

- · and where:

## Tree-shaped Bayesian Networks

A tree-shaped BN over RVs X = ( X i ) d i =1 is a pair ( T , P ), where:

- · T is a directed tree which has RVs X as nodes;
- · P is a collection of distributions p X ( i | X τ ( ) i ), where
- X τ ( ) i is the parent of X i in T ;

and where:

<!-- formula-not-decoded -->

If X i is the root of T then τ ( ) = 0 and i p X ( i | X 0 ) = p X ( i ).

## Tree-shaped Bayesian Networks

A tree-shaped BN over RVs X = ( X i ) d i =1 is a pair ( T , P ), where:

- · T is a directed tree which has RVs X as nodes;
- · P is a collection of distributions p X ( i | X τ ( ) i ), where X τ ( ) i is the parent of X i in T ;

and where:

<!-- formula-not-decoded -->

<!-- image -->

If X i is the root of T then τ ( ) = 0 and i p X ( i | X 0 ) = p X ( i ).

<!-- formula-not-decoded -->

- · We are given a dataset D = { x ( n ) } N n =1 drawn from an unknown distribution p ∗ ( X )

- · We are given a dataset D = { x ( n ) } N n =1 drawn from an unknown distribution p ∗ ( X )
- · We want to learn the 'best' tree-shaped BN ( T , P ) from D

- · We are given a dataset D = { x ( n ) } N n =1 drawn from an unknown distribution p ∗ ( X )
- · We want to learn the 'best' tree-shaped BN ( T , P ) from D
- · In other words, we want to find the best tree-based approximation p ( X ) = d ∏ i =1 p ∗ ( X X i | τ ( ) i ) of p ∗ ( X )

- · Cayley's formula is a result in graph theory named after Arthur Cayley. It states that for every positive integer d , the number of trees on d labeled vertices is d d -2
- · The number of possible trees for any moderate value of d is so enormous as to exlude any approach of exhaustive search

## How many possible trees?

- · Cayley's formula is a result in graph theory named after Arthur Cayley. It states that for every positive integer d , the number of trees on d labeled vertices is d d -2
- · The number of possible trees for any moderate value of d is so enormous as to exlude any approach of exhaustive search

<!-- image -->

We want to find T s.t. its induced probability distribution p ( X ) = ∏ d i =1 p ∗ ( X X i | τ ( ) i ) is as close as possible to the true unknown distribution p ∗ ( X ).

We want to find T s.t. its induced probability distribution p ( X ) = ∏ d i =1 p ∗ ( X X i | τ ( ) i ) is as close as possible to the true unknown distribution p ∗ ( X ).

<!-- formula-not-decoded -->

We want to find T s.t. its induced probability distribution p ( X ) = ∏ d i =1 p ∗ ( X X i | τ ( ) i ) is as close as possible to the true unknown distribution p ∗ ( X ).

<!-- formula-not-decoded -->

We want to find T s.t. its induced probability distribution p ( X ) = ∏ d i =1 p ∗ ( X X i | τ ( ) i ) is as close as possible to the true unknown distribution p ∗ ( X ).

<!-- formula-not-decoded -->

We want to find T s.t. its induced probability distribution p ( X ) = ∏ d i =1 p ∗ ( X X i | τ ( ) i ) is as close as possible to the true unknown distribution p ∗ ( X ).

<!-- formula-not-decoded -->

Since E ∼ x p ∗ [log p ∗ ( x )] is independent of T , only the second quantity matters.

## Chow-Liu Algorithm - The Proof \ 2

<!-- formula-not-decoded -->

## Chow-Liu Algorithm - The Proof \ 2

<!-- formula-not-decoded -->

## Chow-Liu Algorithm - The Proof \ 2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, minimising KL ( p ∗ || p ) is equivalent to maximizing ∑ d i =1 MI( X i , X τ ( ) i ) over all possible trees.

## Maximum spanning tree

Let MI be the Mutual Information matrix of X = ( X i ) 5 i =1 .

<!-- formula-not-decoded -->

## Maximum spanning tree

Let MI be the Mutual Information matrix of X = ( X i ) 5 i =1 .

<!-- formula-not-decoded -->

<!-- image -->

𝑋

.6

4

5

.6

3

𝑋

4

.

67

𝑋

1

𝑋

.7

2

3

𝑋

2

## Maximum spanning tree

Let MI be the Mutual Information matrix of X = ( X i ) 5 i =1 .

<!-- formula-not-decoded -->

<!-- image -->

## Maximum spanning tree

Let MI be the Mutual Information matrix of X = ( X i ) 5 i =1 .

<!-- image -->

- · A maximum spanning tree is a subset of the edges of a connected undirected graph that connects all the vertices together, without any cycles and with the maximum possible total edge weight
- · Kruskal's algorithm finds the maximum spanning tree in polynomial time

## Orienting the Tree

Recall: minimising KL ( p ∗ || p ) is equivalent to maximizing ∑ d i =1 MI( X i , X τ ( ) i ).

Mutual information is symmetric: MI( X i , X τ ( ) i ) = MI( X τ ( ) i , X i )

So direction of the arcs does not impact KL ( p || p

∗ )!

To orient the undirected maximum spanning tree:

- · Choose any node as the root;
- · Orient all edges to point away from the root

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

<!-- formula-not-decoded -->

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Parameter estimation

## A CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

where α &gt; 0 is a smoothing parameter for the Laplace's correction. Usually α = 0 01. .

## Chow-Liu Algorithm

| Algorithm 1 Learn-CLT ( D , α )                                                       |
|---------------------------------------------------------------------------------------|
| Input: A set of samples D = { x ( n ) } N n =1 over RVs X and a smoothing parameter α |
| Output: A CLT ( T , P ) over RVs X 1: MI ← estimateMI( D , α )                        |
| 2: T ← maximumSpanningTree( MI )                                                      |
| 3: T ← directedTree( T )                                                              |
| 4: P ← estimatePMFs( T , D , α )                                                      |
| 5: return ⟨T , P⟩                                                                     |

<!-- image -->

## Chow-Liu Trees:

- · Maximum-likelihood fit to given data over space of tree-shaped BNs
- · Based on maximum-spanning tree for pairwise mutual information
- · Runs in polynomial time using e.g. Kruskal's or Prim's algorithm

## Inference in Tree-shaped BNs

## Suppose we have a tree-shaped BN ( T , P )

- · For instance, a CLT

We now want to perform inference with it:

- · Marginal inference
- · Most Probably Explanation

How can we do this efficiently?

## Consider a CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

Consider a CLT ( T , P ) encoding p ( X ) = p X ( 1 ) p X ( 2 | X 1 ) p X ( 3 | X 5 ) p X ( 4 | X 5 ) p X ( 5 | X 1 ).

<!-- image -->

p x ( 1 = 1 , x 2 = 0 , x 3 = 1 , x 4 = 1 , x 5 = 0) = 0 7 . · 0 6 . · 0 6 . · 0 2 . · 0 4 = 0 02016 . .

## Exhaustive inference: p x ( 2 = 0 , x 5 = 1) = 0 258 .

|   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |
|-------|-------|-------|-------|-------|-----------|-------|-------|-------|-------|-------|-----------|
|     0 |     0 |     0 |     0 |     0 |   0.01728 |     1 |     0 |     0 |     0 |     0 |   0.05376 |
|     0 |     0 |     0 |     0 |     1 |   0.0003  |     1 |     0 |     0 |     0 |     1 |   0.0126  |
|     0 |     0 |     0 |     1 |     0 |   0.00432 |     1 |     0 |     0 |     1 |     0 |   0.01344 |
|     0 |     0 |     0 |     1 |     1 |   0.0003  |     1 |     0 |     0 |     1 |     1 |   0.0126  |
|     0 |     0 |     1 |     0 |     0 |   0.02592 |     1 |     0 |     1 |     0 |     0 |   0.08064 |
|     0 |     0 |     1 |     0 |     1 |   0.0027  |     1 |     0 |     1 |     0 |     1 |   0.1134  |
|     0 |     0 |     1 |     1 |     0 |   0.00648 |     1 |     0 |     1 |     1 |     0 |   0.02016 |
|     0 |     0 |     1 |     1 |     1 |   0.0027  |     1 |     0 |     1 |     1 |     1 |   0.1134  |
|     0 |     1 |     0 |     0 |     0 |   0.06912 |     1 |     1 |     0 |     0 |     0 |   0.03584 |
|     0 |     1 |     0 |     0 |     1 |   0.0012  |     1 |     1 |     0 |     0 |     1 |   0.0084  |
|     0 |     1 |     0 |     1 |     0 |   0.01728 |     1 |     1 |     0 |     1 |     0 |   0.00896 |
|     0 |     1 |     0 |     1 |     1 |   0.0012  |     1 |     1 |     0 |     1 |     1 |   0.0084  |
|     0 |     1 |     1 |     0 |     0 |   0.10368 |     1 |     1 |     1 |     0 |     0 |   0.05376 |
|     0 |     1 |     1 |     0 |     1 |   0.0108  |     1 |     1 |     1 |     0 |     1 |   0.0756  |
|     0 |     1 |     1 |     1 |     0 |   0.02592 |     1 |     1 |     1 |     1 |     0 |   0.01344 |
|     0 |     1 |     1 |     1 |     1 |   0.0108  |     1 |     1 |     1 |     1 |     1 |   0.0756  |

So, we need something smarter

## Variable Elimination

## input:

Bayesian network

variables in network N

- I: ordering of network variables not in Q

output: the marginal Pr(Q) prior

## main:

- 1: S&lt; CPTs of network N
- 2: for i = 1 to length of order T do
- 3: to $ and mentions variable I (i ) belongs
- 5: replace all factors fk in $ by factor fi
- 6: end for
- 7: return f [lfes \_

## Variable Elimination in Trees

## When using reverse topological order on a tree-structured BN:

- · The order width 1 w = 1, so VE will run in O ( n exp ( w )) = O ( n ) time!
- · Algorithm can be elegantly restructured as message passing method

- · Every non-root node sends messages to its parent
- · Every node can send a message if and only if it has received messages from all its children
- · We denote by µ Xi → X τ ( ) i ; x the message sent from X i to its parent X τ ( ) i when X τ ( ) i = x

<!-- image -->

## Marginal Inference: The Sum-Product Algorithm

- · Let ( T , P ) be a tree-shaped BN over X = { X i } d i =1 and X r the root of T
- · We want to compute p (ˆ) where ˆ x x ∈ ˆ X and ˆ X ⊆ X

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Marginal Inference: How to compute p x ( 2 = 0 , x 5 = 1) ?

- · We use X 3 ≻ X 4 ≻ X 2 ≻ X 5 ≻ X 1 as reversed topological order

<!-- image -->

p x ( 2 = 0 , x 5 = 1) = p x ( 1 = 0) · µ X 2 → X 1;0 · µ X 5 → X 1;0 + p x ( 1 = 1) · µ X 2 → X 1;1 · µ X 5 → X 1;1 = 0 3 . · (0 2 . · 0 1) + 0 7 . . · (0 6 . · 0 6) = 0 258 . .

## Marginal Inference: How to compute p x ( 2 = 0 , x 3 = 1 , x 4 = 1) ?

- · We use X 3 ≻ X 4 ≻ X 2 ≻ X 5 ≻ X 1 as reversed topological order

<!-- image -->

p x ( 2 = 0 , x 3 = 1 , x 4 = 1) = p x ( 1 = 0) · µ X 2 → X 1;0 · µ X 5 → X 1;0 + p x ( 1 = 1) · µ X 2 → X 1;1 · µ X 5 → X 1;1 = 0 3 . · (0 2 . · 0 153) + 0 7 . . · (0 6 . · 0 318) = 0 14274 . .

- · The Most Probable Explanation (MPE) task computes the most probable state of variables that do not have evidence
- · The difference between standard inference and MPE inference is that instead of summing values, the maximum is used

## Application: data imputation, e.g. inpainting

<!-- image -->

<!-- image -->

## Exhaustive inference: What is the most probable state?

|   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |
|-------|-------|-------|-------|-------|-----------|-------|-------|-------|-------|-------|-----------|
|     0 |     0 |     0 |     0 |     0 |   0.01728 |     1 |     0 |     0 |     0 |     0 |   0.05376 |
|     0 |     0 |     0 |     0 |     1 |   0.0003  |     1 |     0 |     0 |     0 |     1 |   0.0126  |
|     0 |     0 |     0 |     1 |     0 |   0.00432 |     1 |     0 |     0 |     1 |     0 |   0.01344 |
|     0 |     0 |     0 |     1 |     1 |   0.0003  |     1 |     0 |     0 |     1 |     1 |   0.0126  |
|     0 |     0 |     1 |     0 |     0 |   0.02592 |     1 |     0 |     1 |     0 |     0 |   0.08064 |
|     0 |     0 |     1 |     0 |     1 |   0.0027  |     1 |     0 |     1 |     0 |     1 |   0.1134  |
|     0 |     0 |     1 |     1 |     0 |   0.00648 |     1 |     0 |     1 |     1 |     0 |   0.02016 |
|     0 |     0 |     1 |     1 |     1 |   0.0027  |     1 |     0 |     1 |     1 |     1 |   0.1134  |
|     0 |     1 |     0 |     0 |     0 |   0.06912 |     1 |     1 |     0 |     0 |     0 |   0.03584 |
|     0 |     1 |     0 |     0 |     1 |   0.0012  |     1 |     1 |     0 |     0 |     1 |   0.0084  |
|     0 |     1 |     0 |     1 |     0 |   0.01728 |     1 |     1 |     0 |     1 |     0 |   0.00896 |
|     0 |     1 |     0 |     1 |     1 |   0.0012  |     1 |     1 |     0 |     1 |     1 |   0.0084  |
|     0 |     1 |     1 |     0 |     0 |   0.10368 |     1 |     1 |     1 |     0 |     0 |   0.05376 |
|     0 |     1 |     1 |     0 |     1 |   0.0108  |     1 |     1 |     1 |     0 |     1 |   0.0756  |
|     0 |     1 |     1 |     1 |     0 |   0.02592 |     1 |     1 |     1 |     1 |     0 |   0.01344 |
|     0 |     1 |     1 |     1 |     1 |   0.0108  |     1 |     1 |     1 |     1 |     1 |   0.0756  |

|   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |   x 1 |   x 2 |   x 3 |   x 4 |   x 5 |   p ( x ) |
|-------|-------|-------|-------|-------|-----------|-------|-------|-------|-------|-------|-----------|
|     0 |     0 |     0 |     0 |     0 |   0.01728 |     1 |     0 |     0 |     0 |     0 |   0.05376 |
|     0 |     0 |     0 |     0 |     1 |   0.0003  |     1 |     0 |     0 |     0 |     1 |   0.0126  |
|     0 |     0 |     0 |     1 |     0 |   0.00432 |     1 |     0 |     0 |     1 |     0 |   0.01344 |
|     0 |     0 |     0 |     1 |     1 |   0.0003  |     1 |     0 |     0 |     1 |     1 |   0.0126  |
|     0 |     0 |     1 |     0 |     0 |   0.02592 |     1 |     0 |     1 |     0 |     0 |   0.08064 |
|     0 |     0 |     1 |     0 |     1 |   0.0027  |     1 |     0 |     1 |     0 |     1 |   0.1134  |
|     0 |     0 |     1 |     1 |     0 |   0.00648 |     1 |     0 |     1 |     1 |     0 |   0.02016 |
|     0 |     0 |     1 |     1 |     1 |   0.0027  |     1 |     0 |     1 |     1 |     1 |   0.1134  |
|     0 |     1 |     0 |     0 |     0 |   0.06912 |     1 |     1 |     0 |     0 |     0 |   0.03584 |
|     0 |     1 |     0 |     0 |     1 |   0.0012  |     1 |     1 |     0 |     0 |     1 |   0.0084  |
|     0 |     1 |     0 |     1 |     0 |   0.01728 |     1 |     1 |     0 |     1 |     0 |   0.00896 |
|     0 |     1 |     0 |     1 |     1 |   0.0012  |     1 |     1 |     0 |     1 |     1 |   0.0084  |
|     0 |     1 |     1 |     0 |     0 |   0.10368 |     1 |     1 |     1 |     0 |     0 |   0.05376 |
|     0 |     1 |     1 |     0 |     1 |   0.0108  |     1 |     1 |     1 |     0 |     1 |   0.0756  |
|     0 |     1 |     1 |     1 |     0 |   0.02592 |     1 |     1 |     1 |     1 |     0 |   0.01344 |
|     0 |     1 |     1 |     1 |     1 |   0.0108  |     1 |     1 |     1 |     1 |     1 |   0.0756  |

## MPE Inference: The Max-Product Algorithm

- · Let ( T , P ) be a tree-shaped BN over X = { X i } d i =1 and X r the root of T
- · ˆ x ∈ ˆ , X ˆ X ⊆ X and Z = X \ ˆ X
- · We want to compute max z ∈ Z p (ˆ x , z ) ∝ max z ∈ Z p ( z x | ˆ)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- image -->

max x ∈ X p ( x ) = max[( p x ( 1 = 0) · µ X 2 → X 1;0 · µ X 5 → X 1;0 ) , ( p x ( 1 = 1) · µ X 2 → X 1;1 · µ X 5 → X 1;1 )]

<!-- formula-not-decoded -->

x 1 = 1 = ⇒ x 2 = 0 and x 5 = 1 x 5 = 1 = ⇒ x 3 = 1 and x 4 = 0

<!-- image -->

```
˜ µ X 3 → X 5 ;0 = max[ 4 . ,. 6] = .6 [ [1] ] ˜ µ X 4 → X 5 ;0 = max[ 8 . ,. 2] = .8 [ [0] ] ˜ µ X 2 → X 1 ;0 = 8 . [ [1] ] ˜ µ X 2 → X 1 ;1 = 4 . [ [1] ] ˜ µ X 5 → X 1 ;0 = ( 9 . · .6 · .8 ) = . 432 [ [0] ] ˜ µ X 5 → X 1 ;1 = ( 4 . · .6 · .8 ) = . 192 [ [0] ]
```

max z ∈ Z p (ˆ x , z ) = max[( p x ( 1 = 0) · µ X 2 → X 1;0 · µ X 5 → X 1;0 ) , ( p x ( 1 = 1) · µ X 2 → X 1;1 · µ X 5 → X 1;1 )]

<!-- formula-not-decoded -->

x 1 = 0 and x 2 = 1 and x 5 = 0 x 5 = 0 = ⇒ x 3 = 1 and x 4 = 0

## Inference in Tree-Shaped BNs

<!-- image -->

## Efficient inference with VE using reverse topological order

- · This has order width w = 1, so VE then has complexity O ( n )

Algorithm can be restructured as message passing method

- · Sum-Product algorithm for marginal inference
- · Max-Product algorithm for MPE

## Ancestral Sampling

## Ancestral Sampling

Method to draw i.i.d. samples x ∼ p ( X ), where p ( X ) is (encoded by) a BN ( G P , )

- · Requires method to sample x ∼ p X ( | pa ( X )) for each X
- · E.g. inverse-transform sampling

## Ancestral Sampling

Method to draw i.i.d. samples x ∼ p ( X ), where p ( X ) is (encoded by) a BN ( G P , )

- · Requires method to sample x ∼ p X ( | pa ( X )) for each X
- · E.g. inverse-transform sampling

Simply use topological order π of G . For each i = 1 , . . . , | π | ,

- · Let ρ i = pa ( X π ( ) i )
- · Sample x π ( ) i ∼ p X ( π ( ) i | X ρ i = x ρ i ), where x ρ i are values already sampled

Then x ∼ p ( X )

- · Let X ∼ B ( p ) a Bernoulli RV with probability p . To sample from X we generate a random number ϵ ∈ [0 , 1] if ϵ ≤ p then x = 1 else x = 0.
- · We use X 1 ≺ X 2 ≺ X 5 ≺ X 3 ≺ X 4 as topological order.

<!-- image -->

- 1. rand([0, 1]) = 0.8 → x 1 = 0
- 2. rand([0, 1]) = 0.3 → x 2 = 1
- 3. rand([0, 1]) = 0.5 → x 5 = 0
- 4. rand([0, 1]) = 0.1 → x 3 = 1
- 5. rand([0, 1]) = 0.6 → x 4 = 0

## Summary and Outlook

## Today's lecture

- · Chow-Liu Trees
- · Inference in tree-shaped BNs
- · Ancestral sampling

## Next lecture

- · Markov networks
- · missing data