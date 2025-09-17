import numpy as np
import math 
import random 

#%% modulo 1
def error(x,y):
    return abs(x-y)

def error_relativo(x,y):
    if x == 0 and y == 0:
        return 0.0
    return abs(x-y)/abs(x)

def all_error(a, b):
    res = True 
    for i in range(len(a)):
        if error_relativo(round(a[i]), round(b[i])) > 1e-06: 
            res = False
    return res

def matricesIguales(A, B):
    if len(A) != len(B):
        return False
    res = True
    for i in range(len(A)):
        if (len(A[i]) != len(B[i])):
            res = False 
        else:
            if not all_error(A[i], B[i]):
                    res = False 
    return res 

#%% modulo 2: arreglar trans afin 
def rota(theta): 
    return np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta), math.cos(theta)]])

def escala(s): 
    n = len(s)
    matriz = np.zeros((n, n))
    for j in range(n):
        matriz[j][j] = s[j]
    return np.array(matriz)

def rota_y_escala(theta,s):
    S = escala(s)  
    R = rota(theta)
    
    a = S[0][0]*R[0][0] + S[0][1]*R[1][0]
    b = S[0][0]*R[0][1] + S[0][1]*R[1][1]
    c = S[1][0]*R[0][0] + S[1][1]*R[1][0]
    d = S[1][0]*R[0][1] + S[1][1]*R[1][1]
    
    return np.array([[a, b], [c, d]])

def afin(theta,s,b):
    rt = rota_y_escala(theta, s)
    matriz_afin = np.array([
        [rt[0, 0], rt[0, 1], b[0]],
        [rt[1, 0], rt[1, 1], b[1]],
        [0, 0, 1]
    ])
    return matriz_afin
    
def trans_afin(v,theta,s,b):
    M = afin(theta, s, b)
    v_hom = np.array([v[0], v[1], 1])
   
    x_trans = M[0,0]*v_hom[0] + M[0,1]*v_hom[1] + M[0,2]*v_hom[2]
    y_trans = M[1,0]*v_hom[0] + M[1,1]*v_hom[1] + M[1,2]*v_hom[2]
    
    return [x_trans, y_trans]

#%% funcion inversa 
def inversa(A):
    """
    Calcula la inversa de una matriz cuadrada usando eliminación gaussiana
    Args:
        A: matriz cuadrada (lista de listas)
    Returns:
        Matriz inversa o None si es singular
    """
    n = len(A)
    
    # Verificar que la matriz es cuadrada
    for i in range(n):
        if len(A[i]) != n:
            raise ValueError("La matriz debe ser cuadrada")
    
    # Crear matriz aumentada [A | I]
    aumentada = []
    for i in range(n):
        # Crear una nueva fila con los elementos de A + la identidad
        nueva_fila = []
        # Añadir elementos de A
        for j in range(n):
            nueva_fila.append(float(A[i][j]))
        # Añadir elementos de la matriz identidad
        for j in range(n):
            nueva_fila.append(1.0 if j == i else 0.0)
        aumentada.append(nueva_fila)
    
    # Eliminación gaussiana con pivoteo parcial
    for col in range(n):
        # Pivoteo parcial: encontrar la fila con el mayor valor absoluto en esta columna
        max_fila = col
        max_val = abs(aumentada[col][col])
        
        for i in range(col + 1, n):
            if abs(aumentada[i][col]) > max_val:
                max_val = abs(aumentada[i][col])
                max_fila = i
        
        # Intercambiar filas si es necesario
        if max_fila != col:
            aumentada[col], aumentada[max_fila] = aumentada[max_fila], aumentada[col]
        
        # Verificar si la matriz es singular
        if abs(aumentada[col][col]) < 1e-12:
            return None
        
        # Hacer el elemento diagonal igual a 1
        pivot = aumentada[col][col]
        for j in range(2 * n):
            aumentada[col][j] /= pivot
        
        # Hacer ceros en las otras filas de esta columna
        for i in range(n):
            if i != col:
                factor = aumentada[i][col]
                for j in range(2 * n):
                    aumentada[i][j] -= factor * aumentada[col][j]
    
    # Extraer la matriz inversa (últimas n columnas)
    inversa_mat = []
    for i in range(n):
        fila_inversa = []
        for j in range(n, 2 * n):
            fila_inversa.append(aumentada[i][j])
        inversa_mat.append(fila_inversa)
    
    return inversa_mat



#%% modulo 3
def norma(x, p):
    if p == 'inf': 
        res = 0
        for i in range(len(x)): 
            if res <= abs(x[i]):
                res = abs(x[i])
    elif p == 2: 
        res = 0
        for i in range(len(x)): 
            res += abs(x[i])**2
        res = res ** (1/2)
    elif p == 1: 
        res = 0
        for i in range(len(x)):
            res += abs(x[i]) 
    else: 
        res = 0
        for i in range(len(x)): 
            res += abs(x[i])**p
        res = res ** (1/p) 
    return res 
    
        
def normaliza(X, p): 
    res = []
    for i in range(len(X)):
        n = norma(X[i], p)
        if n == 0:
            res.append(X[i].copy())
        v = []
        for j in range(len(X[i])): 
                v.append(X[i][j] / n)
        res.append(v)
    return res

def traspuesta(A):
    if len(A) == 0:
        return A
        
    res = []
    
    for i in range(len(A[0])):
         fila = []
         for j in range(len(A)): 
              fila.append(A[j][i])
         res.append(fila)
    return res 
    
def normaExacta(A, p=[1,'inf']):  
     if p == 1: 
        n = 0
        for i in range(len(A)): 
                if norma(A[i], 1) >= n:
                    n = norma(A[i], 1)
        return n
         
     elif p == 'inf':
        m = 0
        tras = traspuesta(A)
        for i in range(len(tras)): 
                if norma(tras[i], 1) >= m:
                    m = norma(tras[i], 1)
        return m
     else: 
          return None

def normaMatMC(A, q, p, Np):
    norma_max = 0
    x_max = None
        
    for _ in range(Np):
        x_random = np.random.randn(len(A[0]))
        x_normalizado = x_random / norma(x_random, p)    
        
        Ax = []
        for i in range(len(A)):
            v = 0
            for j in range(len(A[i])):
                v += x_normalizado[j]*A[i][j]
            Ax.append(v)
            
        actual = norma(Ax,q)
        if actual > norma_max:
            norma_max = actual
            x_max = x_normalizado
            
    return norma_max, x_max

def condMC(A, p, Np): # este es con inducida
    res = normaMatMC(A, p, p, Np)[0] * normaMatMC(inversa(A), p, p, Np)[0]
    return res 

def condExacta(A, p): # este es con sumatoria 
    res = normaExacta(A, p) * normaExacta(inversa(A), p)
    return res 
