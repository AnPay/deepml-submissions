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


import numpy as np
from typing import List

'''

'''
class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)
        mean_square = np.mean(x**2,axis=-1,keepdims=True)
        # normalize
        x_hat = (x) / np.sqrt(mean_square+eps)
        output = x_hat*gamma
        return np.round(output,4).tolist()
