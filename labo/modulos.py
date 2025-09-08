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
))
assert(np.allclose(
trans_afin(np.array([1,0]) , np.pi/2,[3,2] ,[4,5]) ,
np.array([0,7])
))
