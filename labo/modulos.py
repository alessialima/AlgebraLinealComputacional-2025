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

# Modulo 4 
def calculaLU(A): 
    U = None 
    nops = 0 
    pivote = 0 
    U = A.copy() 
    L = np.eye(len(A))

    if len(A) != len(A[0]): # si no es cuadrada chau 
        return None, None, 0 
        
    for i in range(len(A)-1): 
        pivote = U[i][i] 
        if abs(pivote) < 1: # 0 < pivote < 1
            return None, None, 0 
            
        for j in range(i + 1, len(U)): 
                coef = U[j][i]/pivote
                U[j][i] = 0 
                nops += 1 
                L[j][i] = coef
                
                for k in range(i+1, len(A)): 
                    U[j][k] -= coef * U[i][k]
                    nops += 2 # mult y resta son dos operaciones 
                    
    return np.array(L), np.array(U), nops

def res_tri(L,b,inferior=True):
    x = len(b)*[0]
    if inferior == True: 
        for i in range(len(b)):
            suma = 0
            for j in range(i):  
                suma += L[i][j] * x[j] # me armo la mini sumita 
            x[i] = (b[i] - suma) / L[i][i]  
        return x
    else: 
        for i in range(len(b)-1, -1, -1):  # cuento abajo a arriba
            suma = 0
            for j in range(i+1, len(b)):  # me armo la sumita 
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
        return x
  
def determinante(A):
    A = np.array(A) 
    
    if len(A) == 0: # escribo casos base 
        return 1 
    if len(A)!=len(A[0]): 
        return 0 
    det = 0
    for i in range(len(A[0])):
            submatriz = []
            for k in range(1, len(A)):
               fila = []
               for l in range(len(A)): 
                   if l != i: 
                       fila.append(A[k][l])
               submatriz.append(fila)
                
            menor = determinante(submatriz)
            signo = (-1)**i 
            det += signo * A[0][i] * menor # menor es la submatriz esa de los det 
    return det
    
def inversa(A):  
            A = np.array(A, dtype=float) 

            if len(A) != len(A[0]): 
                return None 
                
            if abs(determinante(A)) < 1e-12: 
                return None
                
            A_inv = np.eye(len(A))
            A_copia = A.copy()
                
            for i in range(len(A)): 
                if abs(A_copia[i][i]) < 1e-12:
                    res = False 
                    for k in range(i + 1, n): 
                        if abs(A_copia[k][i]) > 1e-12:
                            A_copia[[i, k]] = A_copia[[k, i]]
                            A_inv[[i, k]] = A_inv[[k, i]]
                            res = True 
                            break
                    if not res: 
                          return None 
                
                pivote = A_copia[i][i]
                A_copia[i] = A_copia[i] / pivote
                A_inv[i] = A_inv[i] / pivote
                      
                for j in range(len(A)):

                    if j != i:  
                      coef = A_copia[j][i]  
                      A_copia[j] = A_copia[j] - coef*A_copia[i]
                      A_inv[j] = A_inv[j] - coef*A_inv[i]
            return A_inv

def diagonal(U): # en otra funcion para que sea mas aesthetic 
    U_copia = U.copy()
    D = np.zeros((len(U), len(U)))
    
    for i in range(len(U)):
        pivote = U_copia[i][i]
        
        if abs(pivote) < 1e-12:
            return None  
        
        D[i][i] = pivote
        
        for j in range(i + 1, len(U)):
            factor = U_copia[j][i] / pivote
            for k in range(i, len(U)):
                U_copia[j][k] -= factor * U_copia[i][k]    
    return D

def calculaLDV(A): 
    A = np.array(A, dtype=float) 
    nops = 0
    
    L, U, P = calculaLU(A)
    if L is None or U is None: 
        return None, None, None, nops

    D = diagonal(U) 
    V = U.copy() 
    for i in range(len(A)):
        if abs(V[i, i]) > 1e-08:
            div = V[i][i]
            for j in range(len(V[i])): 
                V[i][j] = V[i][j] / div 
                nops += 1
        else: 
            return None, None, None, nops # es singular 
    return L, D, V, nops  

def esSDP(A, atol = 1e-12): # lo hice mas estricto pq no me pasaba el test 
    A = np.array(A, dtype=float)
    
# es cuadrada?
    if len(A) != len(A[0]): 
        return False 
# es simetrica ? A = A´t ?
    if len(A) != len(A[0]): 
        return False 
    for i in range(len(A)): 
        for j in range(i+1,len(A)): 
          if abs(A[i][j]-A[j][i])>atol:
            return False 
# te odio ldv 
    L, D, V, nops = calculaLDV(A) 
    if L is None or D is None or V is None: 
             return False 
    D = np.array(D)
# diagonales de D positivas ? 
    for l in range(len(D)): 
        if D[l, l] <= atol: 
            return False 
    return True 
