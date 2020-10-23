# einsum2tfop
Convert einsum to tensorflow operation codes.

## What is Einsum?
https://rockt.github.io/2018/04/30/einsum  
**einsum** is notation to express tensor operation in elegant way!  
e.g)  
```
# Matrix multiplication
einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]
# Dot product
einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
# Outer product
einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
# Transpose
einsum('ij->ji', m)  # output[j,i] = m[i,j]
# Trace
einsum('ii', m)  # output[j,i] = trace(m) = sum_i m[i, i]
# Batch matrix multiplication
einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
```
## Requirement
- tensorflow
- tf-coder

## Usage
```python
from einsum2tfop import find_op

find_op('i,i->',[   # <- Dot product vectors.
        tf.constant([1,2,3]) , tf.constant([4,5,6])
   ])
```
this prints :
```
Input 'in1':
tf.Tensor([1 2 3], shape=(3,), dtype=int32)

Input 'in2':
tf.Tensor([4 5 6], shape=(3,), dtype=int32)

Output:
tf.Tensor(32, shape=(), dtype=int32)

Constants: [0, 1, -1, True, False, 3]

Searching...

Found solution: tf.tensordot(in1, in2, 1)

Solution was found in 0.1 seconds:
tf.tensordot(in1, in2, 1)
```
