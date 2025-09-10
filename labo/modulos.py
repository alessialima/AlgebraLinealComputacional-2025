import numpy as np
import math 

#%% Labo 1: funciones 
def error(x,y):
    return abs(x-y)

def error_relativo(x,y):
    return abs(x-y)/abs(x)

def matricesIguales(A, B):
    A_lista = A.tolist()
    B_lista = B.tolist()
    if len(A) != len(B):
        return False
    res = True
    for i in range(len(A)):
        if (len(A[i]) != len(B[i])):
            res = False 
        else:
            for j in range(len(A[i])):
                if not np.allclose(error(A[i],B[i]),0, atol=1e-08):
                    res = False 
    return res 
# Test 
def sonIguales(x,y,atol=1e-08):
    return np.allclose(error(x,y),0,atol=atol)

assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

#%% Labo 2: funciones 
def rota(theta): 
    return np.array([[math.cos(theta), - math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
    

def escala(s): 
    n = len(s)
    matriz = [[0] * n for _ in range(n)]  
    for i in range(n):
        matriz[i][i] = s[i]
    return np.array(matriz)

def rota_y_escala(theta,s):
    return escala(s) @ rota(theta) 

def afin(theta,s,b):
    rt = rota_y_escala(theta,s)
    matriz_afin = np.eye(3)
    matriz_afin[0:2, 0:2] = rt 
    matriz_afin[0:2, 2] = b                 
    
    return matriz_afin
    
def trans_afin(v,theta,s,b):
    v_h = np.array([v[0], v[1], 1])
    r_h =  afin(theta,s,b) @ v_h
    return r_h[:2]

# hay algo mal en el ultimo test pq hay un problema con el redondeo 
# test: 
#Tests para rota
assert(np.allclose(rota(0) , np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1],[1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0] ,[0, -1]])))

#Tests para escala
assert(np.allclose(escala([2,3]) , np.array([[2 ,0] ,[0 ,3]])))
assert(np.allclose(escala([1,1,1]) , np.eye(3)))
assert(
np.allclose(escala([0.5,0.25]) , np.array([[0.5 ,0] ,[0 ,0.25]]))
)

#Tests para rota y escala
assert(
np.allclose(rota_y_escala(0,[2,3]) , np.array([[2 ,0] ,[0 ,3]]))
)
assert(np.allclose(
rota_y_escala(np.pi/2,[1,1]) , np.array([[0,-1] ,[1,0]])
))
assert(np.allclose(
rota_y_escala(np.pi ,[2,2]) , np.array([[-2,0] ,[0,-2]])
))

#Tests para afin
assert(np.allclose(
afin(0,[1,1] ,[1,2]) ,
np.array([[1,0,1] ,
[0,1,2] ,
[0,0,1]])))

assert(np.allclose(afin(np.pi/2,[1,1] ,[0,0]) ,
np.array([[0,-1,0] ,
[1, 0,0] ,
[0, 0,1]])))

assert(np.allclose(afin(0,[2,3] ,[1,1]) ,
np.array([[2,0,1] ,
[0,3,1] ,
[0,0,1]])))
 
#Tests para trans afin
assert(np.allclose(
trans_afin(np.array([1,0]) , np.pi/2,[1,1] ,[0,0]) ,
np.array([0,1])
))
assert(np.allclose(
trans_afin(np.array([1,1]) , 0,[2,3] ,[0,0]) ,
np.array([2,3])

#%% Labo 3 
def norma(x, p):
    if isinstance(x, (int, float, complex)):
        return abs(x)
    if len(x) == 0:
        return 0
    if p == float('inf') or p == 'inf': # infinito whats 
        return max(abs(val) for val in x)
    res = 0
    for i in range(len(x)):
        res += (abs(x[i])**p)
    return (res**(1/p))

def normaliza(X, p):
    res = []
    for i in range(len(X)):
        n = norma(X[i],p)
        if n == 0:
            normalized_vec = X[i].copy()  
        else:
            normalized_vec = X[i] / n
        res.append(normalized_vec)
    return res

# lo robe de deepseek pq wtf
def normaMatMC(A, q, p, Np):
    m, n = A.shape
    max_norm = 0
    best_x = None
    if q == 'inf':
        q = np.inf 
    if p == 'inf':
        p = np.inf
    
    for i in range(Np):
        x_random = np.random.randn(n)        
        if p == float('inf'):
            x_normalized = x_random / np.max(np.abs(x_random))
        else:
            norm_p = np.linalg.norm(x_random, p)
            x_normalized = x_random / norm_p
            
        Ax = A @ x_normalized
        norm_q = np.linalg.norm(Ax, q)
        
        if norm_q > max_norm:
            max_norm = norm_q
            best_x = x_normalized.copy()
            
    return max_norm, best_x

def normaExacta(A,p=[1,'inf']):
    result = {}
    if 1 in p:
        result[1] = np.max(np.sum(np.abs(A), axis=0))
    if 'inf' in p:
        result['inf'] = np.max(np.sum(np.abs(A), axis=1))
    return result

def condMC(A, p):
    A_inv = np.linalg.inv(A)
    return normaMatMC(A, p, p, 10000) * normaMatMC(A_inv, p, p, 10000)

def condExacto(A, p):
    if p == 'inf':
        p = np.inf
    norma_A = normaExacta(A, p)
    A_inv = np.linalg.inv(A)
    norma_inv = normaExacta(A_inv, p)
    
    return norma_A * norma_A_inv
#%% Test
# Test Norma 
assert(np.allclose(norma(np.array([1,1]), 2), np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

# Test normaliza
assert([np.allclose(norma(x,2) ,1) for x in normaliza([np.array([1]*k) for k in range(1,11)] ,2) ])
assert([not np.allclose(norma(x,2) ,1) for x in normaliza([np.array([1]*k) for k in range(1,11)] ,1) ])
assert([np.allclose(norma(x, 'inf') ,1) for x in normaliza([np.random.rand(k) for k in range(1,11)] , 'inf') ])

# Test normaMC
nMC=normaMatMC(A=np.eye(2) ,q=2,p=1,Np=100000)
assert(np.allclose(nMC[0] ,1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]) ,1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]) ,1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]) ,0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]) ,0,atol=1e-3))

nMC=normaMatMC(A=np.eye(2) ,q=2,p='inf' ,Np=100000)
assert(np.allclose(nMC[0] ,np.sqrt(2) ,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]) ,1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]) ,1,atol=1e-3))

A=np.array([[1 ,2] ,[3 ,4]])
nMC=normaMatMC(A=A,q='inf',p='inf' ,Np=1000000)
assert(np.allclose(nMC[0] ,normaExacta(A,'inf'),rtol=2e-1))

# Test Norma Exacta 
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]) ,1) ,2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]) ,1) ,7))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]) , 'inf') ,6))
assert(normaExacta(np.array([[1,-2],[-3,-4]]) ,2) is None)
assert(normaExacta(np.random.random((10,10)) ,1)<=10)
assert(normaExacta(np.random.random((4,4)), 'inf')<=4)

# Test condMC 
A = np.array([[1 ,1] ,[0 ,1]])
A = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA = normaMatMC(A ,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA[0] ,condA,atol=1e-3))

A=np.array([[3 ,2] ,[4 ,1]])
A =np.linalg.solve(A,np.eye(A.shape[0]))
normaA=normaMatMC(A,2,2,10000)
normaA =normaMatMC(A ,2,2,10000)
condA=condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA[0] ,condA,atol=1e-3))

# Test condExacta 
A=np.random.rand(10,10)
A =np.linalg.solve(A,np.eye(A.shape[0]))
normaA=normaExacta(A,1)
normaA =normaExacta(A ,1)
condA=condExacta(A,1)
assert(np.allclose(normaA*normaA ,condA))

A=np.random.rand(10,10)
A =np.linalg.solve(A,np.eye(A.shape[0]))
normaA=normaExacta(A, 'inf')
normaA =normaExacta(A , 'inf')
condA=condExacta(A, 'inf')
assert(np.allclose(normaA*normaA ,condA))

))
assert(np.allclose(
trans_afin(np.array([1,0]) , np.pi/2,[3,2] ,[4,5]) ,
np.array([0,7])
))
