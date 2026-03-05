import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    n = len(y)
    values, counts = np.unique(y, return_counts=True)
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
    
