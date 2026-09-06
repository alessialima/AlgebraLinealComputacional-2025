import numpy as np 

#%% norma 

def norma(x, p): 
    if p == 'inf': 
        res = 0
        for i in range(len(x)): 
            if res <= abs(x[i]):
                res = abs(x[i])
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

#%% normaliza 

def normaliza(X, p): 
    res = []
    for i in range(len(X)):
        n = norma(X[i], p)
        if n == 0:
            res.append(0)
        else:
            v = []
            for j in range(len(X[i])): 
                v.append(X[i][j] / n)
            res.append(v) 
    return res

#%% norma exacta 

def normaExacta(A,p=[1,'inf']):
    if p!=[1,'inf'] and p!=['inf',1] and p not in [1,'inf']:  
        return None
    norma1 = 0
    normainf = 0
    filas = np.shape(A)[0]
    columnas = np.shape(A)[1]
    maximo1= 0
    sumainf= 0
    maximoinf= 0
    for j in range(columnas):
        suma1= 0
        for i in range(filas):
            suma1+= abs(A[i,j])
            if suma1 > maximo1:
                maximo1 = suma1
    norma1= maximo1
    for i in range(filas):
        sumainf= 0
        for j in range(columnas):
            sumainf+= abs(A[i,j])
            if sumainf > maximoinf:
                maximoinf= sumainf 
    normainf= maximoinf

    return [norma1,normainf]

#%% normaMatMC (no pasa el test !!) 

def norma_filas(M, p, cant):
    if p == 'inf':
        res = np.zeros(cant) 
        for j in range(np.shape(M)[1]):
            col = abs(M[:, j]) 
            res = (res + col + abs(res - col)) / 2 
        return res 
    elif p == 1:
        res = np.zeros(cant) 
        for j in range(np.shape(M)[1]):
            res += abs(M[:, j])**p 
        return res ** (1/p) 
  # elif p is None:
 #      return None 
    else: 
        res = np.zeros(cant) 
        for j in range(np.shape(M)[1]):
            res += abs(M[:, j])**p
        return res ** (1/p) 

def normaMatMC(A, q, p, Np):

    A = np.array(A)
    m, n = np.shape(A) 

    X = np.random.randn(Np, n) 
    normas_x = norma_filas(X, p, Np) 
    X = X / normas_x.reshape(Np, 1) 

    Ax = np.zeros((Np, m)) 
    for j in range(n):
        col_A = A[:, j]
        col_X = X[:, j] 
        Ax += col_X.reshape(Np, 1) * col_A.reshape(1, m) 
    
    normas_ax = norma_filas(Ax, q, Np) 

    norma_max = normas_ax[0]
    idx_max = 0 
    for k in range(Np):
        if normas_ax[k] > norma_max:
            norma_max = normas_ax[k]
            idx_max = k 

    x_max = X[idx_max, :]
    return norma_max, x_max

#%% condExacto 

def condExacta(A, p): 
    res = normaExacta(A, p) * normaExacta(np.linalg.inv(A), p) 
    return res 

#%% condMC 

def condMC(A, p):
    Np = 100000
    res = normaMatMC(A, p, p, Np)[0] * normaMatMC(np.linalg.inv(A), p, p, Np)[0]
    return res 