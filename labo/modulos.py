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

# modulo 5 
def norma2(a):
    suma = 0
    for i in range(len(a)):
        suma += (abs(a[i])**2)
    return (suma)**(1/2)

def producto(a, B):
    res = [0]*len(B) 
    for i in range(len(B)): 
        res[i] = B[i] * a 
    return res 

def producto_escalar(A, B):
    res = 0
    for i in range(len(A)): 
        res += (A[i]*B[i])
    return res 

def resta_vectores(v, w):
    res = [0]*len(v) 
    for i in range(len(res)): 
        res[i] = v[i] - w[i] 
    return res 
    

def QR_con_GS(A, tol = 1e-12, retorna_nops = False): 
    A = np.array(A, dtype= float) # por las dudas 
    nops = 0

    if len(A) != len(A[0]): # SOLO puede resolverse con matrices cuadradas 
        Q = None 
        R = None 
        nops = None 
    
    n = len(A)    
    Q = np.zeros((n,n))
    R = np.zeros((n,n)) 
    
    At = traspuesta(A) 
    
    for j in range(n):
        R[j][j] = norma2(At[j])
        nops += n
        
        if R[j][j] < tol:           
            for i in range(n):
                Q[i][j] = 1.0 if i == j else 0.0 # si no cumple tolerancia lo agrandamos un cachitin. no quiero 0 o cercanos 
        else:
           
            for i in range(n):
                Q[i][j] = At[j][i] / R[j][j]
            nops += n
            
        for k in range(j+1, n):
            columna_q = [Q[i][j] for i in range(n)] 
            R[j][k] = producto_escalar(columna_q, At[k])
            nops += n
            
           
            proj = producto(R[j][k], columna_q) 
            At[k] = resta_vectores(At[k], proj)
            nops += n
    
    if retorna_nops:
        return Q, R, nops
    else:
        return Q, R  
        
#%% HouseHolder 
def multiplicarMatrices(A, B):
    m = len(A)
    n = len(B[0])
    p = len(B)
    res = np.zeros((n,m))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                res[i][j] += A[i][k] * B[k][j]
    return res

def resta_matrices(A, B):
    res = []
    for i in range(len(A)):
        fila = []
        for j in range(len(A[i])):
            fila.append(A[i][j] - B[i][j])
        res.append(fila)
    return res
            
def productoExterior(x, y):
    m = len(x)
    n = len(y)
    res = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            res[i][j] = x[i] * y[j]
    return res

def multiplicar_escalar_vector(escalar, vector):
    return [escalar * elemento for elemento in vector] #deepseek 

def matriz_identidad(m):
    res = np.zeros((m, m))
    for i in range(len(res)): 
        res[i][i] = 1
    return res 
    
def multiplicar_escalar_matriz(escalar, matriz):
    return [[escalar * elemento for elemento in fila] for fila in matriz] # deepseek
    
def QR_con_HH(A,tol=1e-12):
    A = np.array(A, dtype= float)
    
    
    m = len(A) 
    n = len(A[0])
    
    if m < n:
             return None, None 
    
    Q = matriz_identidad(m)
    R = A.tolist()
                
    for k in range(n):
        
        x = []
        for i in range(k, m):
            x.append(R[i][k])
            
        norma_x = norma2(x)
        if norma_x < norma2(x):
            continue
            
        if x[0] >= 0: 
            alpha = -norma_x # quiero que sea negativo, no cambiarle el signo (segun ds)
        else:
            alpha = norma_x
            
        e1 = [0] * len(x) # la canonica e1
        e1[0] = 1

        alpha_e1 = multiplicar_escalar_vector(alpha, e1)
        u = resta_vectores(x, alpha_e1) # hallando u 

        norma_u = norma2(u)
        if norma_u > tol: 
            u_norm = multiplicar_escalar_vector(1/norma_u, u)

            # quiero armar hkmoño
            I = matriz_identidad(p) 
            uuT = productoExterior(u_norm, u_norm)
            uuT_2 = multiplicar_escalar_matriz(2, uuT)
            Hk = resta_matrices(I, uuT_2)
            
           
            Hk_ = matriz_identidad(m)
            for i in range(k, m):
                for j in range(k, m):
                    Hk_[i][j] = Hk[i-k][j-k]
            
            R = multiplicarMatrices(Hk_, R) 
            Q = multiplicarMatrices(Q, traspuesta(Hk_)) 
    return np.array(Q), np.array(R)
                
#&& calcula QR    

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH':
        return QR_con_GS(A, tol, False)
    else:
        return QR_con_HH(A, tol)
#&& Modulo 6 

def multiplicarMv(A, v):
    n = len(A)
    resultado = [0] * n
    for i in range(n):
        for j in range(len(v)):
            resultado[i] += A[i][j] * v[j]
    return resultado

def productoInterno(a,b):
    res = 0
    for i in range(len(b)):
        res += a[i] * b[i]
    return res

def divisionVectorEscalar(a, b): 
    for i in range(len(a)): 
        a[i] = a[i] / b 
    return a 

def fAv(A, v): # fA2(v) 
    Av1 = multiplicarMv(A, v)
    v1 = divisionVectorEscalar(Av1,norma2(Av1))
    Av2 = multiplicarMv(A, v1) 
    return divisionVectorEscalar(Av2,norma2(Av2))

def metpot2k(A, tol=1e-15,K=1000):
    A = np.array(A, dtype= float)
    n = len(A)
    
    v = [0]*n
    for i in range(n): 
        v[i] = random.random()

    v_moño = fAv(A, v)
    e = productoInterno(v_moño,v)
    k = 0
    
    while abs(e-1)>tol and k<K:
        v = v_moño.copy()
        v_moño = fAv(A, v)
        e = productoInterno(v_moño, v)
        k += 1
    
    Av = multiplicarMv(A, v_moño)
    lambd = productoInterno(Av, v_moño) 
        
    epsilon = e - 1
    return v, lambd, epsilon 
        
 #%% 

def resta_vectores(v, w): 
    res = [0]*len(v)
    for i in range(len(v)):
        res[i] = v[i] - w[i]
    return res 
        

def diagRH(A, tol=1e-15, K=1000): 
    v, lamd, epsilon = metpot2k(A,tol, K)
    e = [0]*len(v) 
    e[0] = 1 
    e_1 = resta_vectores(v, e)
    prodExt = productoExterno(e_1, e_1) 
    

