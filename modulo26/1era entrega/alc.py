##% Módulo 1: Error numérico

import numpy as np 

def error (x,y):
    res = np.float64(0)
    x_64 = np.float64(x)
    y_64 = np.float64(y)
    res = np.abs(x_64 - y_64)
    return res

def error_relativo (x,y):
    x_64 = np.abs (np.float64(x))
    if x_64 == 0:
        return 0
    return (error (x,y) / x_64)

def matricesIguales(A,B):
  res: bool = True
  filas = A.shape[0]
  columnas = A.shape[1]
  tolerancia = 1e-7
  i = 0
  j = 0
  if (A.shape != B.shape):     
      res = False
  else:
      while(res == True and i <filas ):
        while (res == True and j <columnas):
            if abs(A[i, j] - B[i, j]) > tolerancia:
                res = False 
            j+=1
        i+= 1
  return res

##% Módulo 2: Transformaciones lineales

def rota(theta):
  return np.array([[np.cos(theta), - np.sin(theta)],[np.sin(theta), np.cos(theta)]])

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
