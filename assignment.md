# This assignment is to predict whether a node pair (node1, node2) has hidden relation (i.e., hidden edge).
- There are about 186,000 edges provided for you to reconstruct the social network and the training dataset. This is a directed network, so each node pair represents a directed edge. E.g., (38751, 38824) represents an edge from node 38751 to node 38824 .
- The social network has about 33,000 hidden edges. These are the relationships you are asked to predict. You can use any prediction method, and you can use any functions/libraries/packages directly.
- In your report, please briefly describe the algorithm you use, and provide instructions about how to execute your program.

# Your Task
- This is a Directed Graph prediction task.
- You will receive a training set, train.csv, which contains the known edges of the network (i.e., Node1 follows Node2). Your task is to use this training data to construct graph features or train a machine learning model to predict whether a hidden follow relationship exists between the node pairs (Node1, Node2) in the test set, test.csv.

# Evaluation Metric
The evaluation metric for this assignment is the ROC AUC (Area Under the Receiver Operating Characteristic Curve).