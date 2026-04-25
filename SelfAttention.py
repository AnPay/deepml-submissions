'''
Implement Self-Attention Mechanism
https://www.deep-ml.com/problems/53?from=Attention%20Is%20All%20You%20Need
'''
import numpy as np
def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    #q,k,v = compute_qkv(Q,K,V)
    
    d_k = Q.shape[-1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    # stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_score = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    attention_output = attention_score @ V
    return attention_output
