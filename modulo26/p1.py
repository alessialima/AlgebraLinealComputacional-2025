"""
Practica 1: 
"""

import numpy as np 

def error(x,y): 
  return abs(x-y)

def error_relativo(x,y): 
  if x == 0: 
    return error(x,y) # no me acuerdo si era asi o dejabamos q se rompa y ya fue 
  return abs(x-y)/abs(x) 

def matricesIguales(A, B): 
  # paso 1: miro las dimensiones 
  a, b = A.shape 
  m, n = B.shape 
  if a != m or b != n: 
     return False 
  # paso 2: miro los elementos 
  for i in range(a): 
    for j in range(b): 
      if error(A[i][j], B[i][j]) > 1e-08: 
        return False 
  return True 

""" 
Practica 2
"""
