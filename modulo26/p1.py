"""
Practica 1: 
"""

import numpy as np 

def error(x,y): 
  x = np.float64(x)
  y = np.float64(x)
  return abs(x-y)

def error_relativo(x,y): 
  x = np.float64(x)
  y = np.float64(x)
  if x == 0: 
    return error(x,y) 
  return abs(x-y)/abs(x) 

def matricesIguales(A, B): 
  A = np.array(A, dtype=np.float64)
  B = np.array(B, dtype=np.float64)

  a, b = A.shape 
  m, n = B.shape 
  if a != m or b != n: 
     return False 
    
  for i in range(a): 
    for j in range(b): 
      if error(A[i][j], B[i][j]) > 1e-08: 
        return False 
  return True 

""" 
Practica 2
"""








