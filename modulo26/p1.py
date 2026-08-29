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

def rota(theta):
  return np.array([[np.cos(theta), - np.sin(theta)],[np.sin(theta), np.cos(theta)]]

def escala(s): 
  n = len(s) 
  res = []
  for i in range(n):
    fila = []
    for j in range(n):
      if i != j:
        fila.append(np.float64(0))
      else: 
        fila.append(s[j])
    res.append(fila)
  res = np.array(res, dtype=np.float64)
  return res 

def rota_y_escala(theta, s):
  S = escala(s)
  R = rota(theta)
  a = S[0,0]*R[0,0] + S[0,1]*R[1,0]
  b = S[0,0]*R[0,1] + S[0,1]*R[1,1]
  c = S[1,0]*R[0,0] + S[1,1]*R[1,0]
  d = S[1,0]*R[0,1] + S[1,1]*R[1,1]
  return np.array([[a,b],[c,d]], dtype=np.float64) 

def afin(theta, s, b): 
  rt = rota_y_escala(theta, s) 
  matriz_afin = np.array([[rt[0,0],rt[0,1],b[0]],[rt[1,0],rt[1,1],b[1]],[0,0,1]], dtype=np.float64) 
  return matriz_afin 

def trans_afin(v, theta, s, b): 
  matriz_afin = afin(theta,s,b) 
  v_hom = np.array([v[0],v[1],1], dtype=np.float64) 
  x = matriz_afin[0,0]*v_hom[0]+matriz_afin[0,1]*v_hom[1]+matriz_afin[0,2]*v_hom[2]
  y = matriz_afin[1,0]*v_hom[0]+matriz_afin[1,1]*v_hom[1]+matriz_afin[1,2]*v_hom[2]
  return np.array([x,y], dtype=np.float64) 
  




