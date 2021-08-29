
## A Brief Intro To Graph Neural Networks
*A brief, very brief, introduction to the hot field of graph neural networks.*

Deep Learning has revolutionized machine learning on all types of tasks ranging from computer vision to natural language processing or sequence modeling. Most of these applications however involve mostly euclidean data that are constrained to some fixed dimensions.

What happens when your data is of non-euclidean nature? Graphs are one way to represent such non-euclidean data, which represent it in form of objects linked with each other through relationships. Machine learning using graphs has always been around, however with the advances in deep learning, recently there have been some exciting developments for learning on graphs.

![image source: stanford cs224w](https://shindeshu.github.io/assets/images/euclidean.png)

<img src="https://shindeshu.github.io/assets/images/euclidean.png" width="200" />


What is a graph, you say? Graph is a set of vertices / nodes (our objects of interest), with edges (relationships between our objects). For example in a social media graph, an account would be a node, and them following someone could be an edge. Numerically, a graph can be represented as a matrix (adjacency), or as a list (of edges).

What data can be represented in the form of graphs? A lot of it! Interactions on a social media site, financial transactions, citation networks, molecules, all these can be represented in the form of graphs and can then be leveraged for machine learning.

Graph representation learning: when we do have a graph (i.e. our nodes, their features, their edges, *their* features), our objective is to learn embeddings for each node, such that two "similar" nodes will have their embeddings closer in space. This embedding for a node should bake into itself its relationships and its neighbourhood and their features (apart from its own). This embedding vector can then be used for our downstream tasks.

![image source: stanford cs224w](https://shindeshu.github.io/assets/images/node_rep_learning.png)

Learning the embedding: while there are many ways to skin this particular cat, the one that's hot right now is called "message passing" or a graph convolution layer. The core concept is pretty simple. Lets say our current node of interest, has three neighbours. Each one of these will pass a "message" to our node, this message being the current state of the node. These messages will be aggregated together with our node's current state, and this will be used to update the node's state to next state. After covering for all nodes, you'd get a complete pass over the entire graph, for a single graph convolution layer. Different frameworks will have different ways of passing messages, or updating them, but the underlying principle is pretty same.

The details of message passing, we'll go over in another post- since this is supposed to be a "brief" introduction.
