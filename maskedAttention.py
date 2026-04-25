#https://www.deep-ml.com/problems/107?from=Attention%20Is%20All%20You%20Need
import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	d_k=Q.shape[-1]
	score = Q@K.T/np.sqrt(d_k)
	score = score + mask
	score = score-np.max(score,axis=-1,keepdims=True)
	attn_score = np.exp(score)/(np.sum(np.exp(score),axis=-1,keepdims=True))
	sel_attn = attn_score@V
	return sel_attn
	
