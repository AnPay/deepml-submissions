#https://www.deep-ml.com/problems/109?from=Attention%20Is%20All%20You%20Need

import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	mean = np.mean(X,axis=-1, keepdims=True)
	var = np.var(X,axis=-1, keepdims=True)
	# normalize
	x_hat = (X - mean) / np.sqrt(var + epsilon)

    # scale and shift
	out = gamma * x_hat + beta
	return out
