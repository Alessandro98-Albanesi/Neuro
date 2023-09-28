import itertools
import numpy as np


def Kabsch_Algorithm (A,B):

    
    N = A.shape[1]
    
   
    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))
    
    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid
    
    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose
    
    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t, A_centroid




Obj_points = [[0,0,0],[0,-0.045,-0.05],[0,-0.095,-0.025],[0,-0.145,-0.025]]
Measured_points = [[0.08711975950547886, 0.011829028009062895, 0.1094], [-0.01220834281863028, -0.032822663950594, 0.0974], [-0.23544135581270434, -0.07522863791400031, 0.1294], [-0.08877378583737128, -0.12684970667811743, 0.0934]]

permuted_list = list(itertools.permutations(Measured_points))

X = np.transpose(np.array(Measured_points))
Y = np.transpose(np.array(Obj_points))
min_err = 1000



for iter in range(len(permuted_list)):
    P = np.transpose(np.array(permuted_list[iter]))
    print(P)
    Rot, Transl, centroid = Kabsch_Algorithm (P,Y)
    
    error = np.linalg.norm(Y - Rot @ P- Transl, 'fro')
    C = Rot @ P + Transl
    print(C)
    
    if error < min_err:
        min_err = error
        match = P
        match_R = Rot
        match_t = Transl


