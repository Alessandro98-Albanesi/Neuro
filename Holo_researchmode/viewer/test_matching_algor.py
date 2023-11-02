import itertools
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import MerweScaledSigmaPoints as SigmaPoints
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def Kabsch_Algorithm (A,B):

    
    N = A.shape[1]
    
    T = np.array([[0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,1]])
   
    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))
    
    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid
    
    # rotation estimation
    H = np.matmul(A_prime, B_prime.transpose())
    svd = np.linalg.svd(H)

    # Replace Arun Equation 13 with Fitzpatrick, chapter 8, page 470,
    # to avoid reflections, see issue #19
    X = _fitzpatricks_X(svd)

    # Arun step 5, after equation 13.
    det_X = np.linalg.det(X)

    if det_X < 0 and np.all(np.flip(np.isclose(svd[1], np.zeros((3, 1))))):

        # Don't yet know how to generate test data.
        # If you hit this line, please report it, and save your data.
        raise ValueError("Registration fails as determinant < 0"
                         " and no singular values are close enough to zero")

    if det_X < 0 and np.any(np.isclose(svd[1], np.zeros((3, 1)))):
        # Implement 2a in section VI in Arun paper.
        v_prime = svd[2].transpose()
        v_prime[0][2] *= -1
        v_prime[1][2] *= -1
        v_prime[2][2] *= -1
        X = np.matmul(v_prime, svd[0].transpose())

    # Compute output
    R = X
    t = B_centroid - R @ A_centroid
    T[0:3,0:3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]
    return R, t, T


def _fitzpatricks_X(svd):
    """This is from Fitzpatrick, chapter 8, page 470.
       it's used in preference to Arun's equation 13,
       X = np.matmul(svd[2].transpose(), svd[0].transpose())
       to avoid reflections.
    """
    VU = np.matmul(svd[2].transpose(), svd[0])
    detVU = np.linalg.det(VU)

    diag = np.eye(3, 3)
    diag[2][2] = detVU

    X = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
    return X

def compute_fre(fixed, moving, rotation, translation):
    """
    Computes the Fiducial Registration Error, equal
    to the root mean squared error between corresponding fiducials.

    :param fixed: point set, N x 3 ndarray
    :param moving: point set, N x 3 ndarray of corresponding points
    :param rotation: 3 x 3 ndarray
    :param translation: 3 x 1 ndarray
    :returns: Fiducial Registration Error (FRE)
    """
    # pylint: disable=assignment-from-no-return

    transformed_moving = np.matmul(rotation, moving) + translation
    squared_error_elementwise = np.square(fixed
                                          - transformed_moving)
    square_distance_error = np.sum(squared_error_elementwise, 1)
    sum_squared_error = np.sum(square_distance_error, 0)
    mean_squared_error = sum_squared_error / fixed.shape[0]
    fre = np.sqrt(mean_squared_error)
    return fre

  
    
def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    N = A.shape[1]
    assert B.shape[1] == N

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
    
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]

    return R, t, T




Obj_points = [[0,0,0], [0,-0.045,-0.05],[0,-0.09,-0.025], [0,-0.14,-0.025]]
Measured_points = [[-0.04990941286087036, -0.018420934677124023, 0.1817748099565506], [-0.03286895528435707, 0.02641139179468155, 0.18464677035808563], [-0.09134603291749954, -0.06017249450087547, 0.1785595864057541], [-0.052152495831251144, -0.1053008958697319, 0.17332105338573456]]
permuted_list = list(itertools.permutations(Measured_points))

X = np.transpose(np.array(Measured_points))
Y = np.transpose(np.array(Obj_points))
min_err = 1000



for iter in range(len(permuted_list)):

    P = np.transpose(np.array(permuted_list[iter]))
    #print(P)
    Rot, Transl, T= arun (P,Y)
    
    error = np.linalg.norm(Rot @ P + Transl - Y, 'fro')
    #print(error)
    
    if error < min_err:
        min_err = error
        match = P
        match_R = Rot
        match_t = Transl
        match_T = T

#result = KF_based_registration(match,Y,match_R)

#print(match)
#print(result)
#print(match_R)
#print(min_err)
print(match)
C = match_R @ match + match_t
print(C)




def compute_R(theta_x, theta_y, theta_z):
        
        
        Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
        ])

        R = np.dot(Rx, np.dot(Ry, Rz))

        return R



U = np.random.uniform(-0.250, 0.250, size=(3, 100))

rotation_angles = np.random.uniform(-90, 90, size=(3,))


translation = np.random.uniform(-0.090, 0.090, size=(3,))


R_gener = compute_R(rotation_angles[0], rotation_angles[1], rotation_angles[2])
t_gener = translation


E = R_gener @ U + t_gener.reshape(-1,1)

covariance = 0.001 * np.eye(3)  # Identity matrix representing the covariance matrix
noise = np.random.multivariate_normal(mean=np.zeros(3), cov=covariance, size=U.shape[1]).T


U_noisy = U + noise



def compute_sigma_points(x, P, alpha, beta, lambda_):
    """
    Compute sigma points for a 6x1 vector.

    Parameters:
    - x: 6x1 mean vector
    - P: 6x6 covariance matrix
    - alpha, beta, kappa: Parameters for controlling the spread of sigma points

    Returns:
    - sigma_points: Array of sigma points, each column represents a sigma point
    """
    n = len(x)  # Dimension of the state vector 

    # Compute the sigma points
    sigma_points = np.zeros((n, 2 * n + 1))
    # Compute sigma points as columns of matrix X
    #sqrt_P = np.linalg.cholesky(P)
    V,S,V_t = np.linalg.svd(P)
    sqrt_S = np.diag(np.sqrt(S))
    sqrt_P = V @ sqrt_S @ V_t


    sigma_points[:, 0] = x  # Mean as the first sigma point

    for i in range(n):
    
        sigma_points[:, i + 1] = x + np.sqrt((n + lambda_))*sqrt_P[:, i]
        sigma_points[:, n + i + 1] = x - np.sqrt((n + lambda_))*sqrt_P[:, i]

    
    
    return sigma_points


def find_closest_point(matrix1, matrix2):
    
    N = matrix2.shape[1]

    # Calculate the Euclidean distance between each point in matrix2 and all points in matrix1
    distances = cdist(matrix2.T, matrix1.T, metric='euclidean')

    # Initialize arrays to store results
    assigned_indices = set()
    closest_points_matrix1 = np.zeros((3, N))
    indices = []
    # Find unique closest points
    for i in range(N):
        min_distance_index = np.argmin(distances[i])
        
        # Check if the closest point has already been assigned
        while min_distance_index in assigned_indices:
            # Set the distance of the already assigned point to infinity
            distances[i, min_distance_index] = np.inf
            min_distance_index = np.argmin(distances[i])
        
        # Assign the closest point to the set of assigned indices
        assigned_indices.add(min_distance_index)
        
        # Update the closest points matrix
        closest_points_matrix1[:, i] = matrix1[:, min_distance_index]
        indices.append(min_distance_index)
    #print(indices)
    return closest_points_matrix1



def UKF(U_init, Y):
        
        x_k_posterior = np.array([0,0,0,0,0,0])
        P_k_posterior = np.identity(6)
        sigma_x = 0.001  # Example value, adjust as needed
        sigma_y = 0.001  # Example value, adjust as needed
        sigma_z = 0.001  # Example value, adjust as needed
        # Create covariance matrix
        covariance_matrix_noise = np.diag([sigma_x, sigma_y, sigma_z])
        variances_fixed_points = np.var(Y, axis=1)
        w_mean = np.zeros(13)
        Wc = np.zeros(13)
        alpha = 1
        k = 3
        lambda_ = alpha**2 * (6 + k) - 6
        beta = 2
        w_mean[0] = lambda_ / (6 + lambda_)
        w_mean[1:13] = 1 / (2 * (6 + lambda_))
        Wc[0] = w_mean[0] + (1 - alpha**2 + beta)
        Wc[1:13] = w_mean[1:13]    
        covariance_of_process_model = np.diag([sigma_x,sigma_y,sigma_z,180/((np.sqrt(variances_fixed_points[1]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_y)))),180/((np.sqrt(variances_fixed_points[0]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_x)))),180/((np.sqrt(variances_fixed_points[0]/sigma_y)+(np.sqrt(variances_fixed_points[1]/sigma_x))))]) 
        U = U_init
        treshold = 0.001
        fre = 1

        print("lambda", lambda_)
        print("w_mean",w_mean)
        print("Wc",Wc)
        print(np.sum(w_mean))

        

        while(fre > treshold):
            #previously_selected_points = np.zeros((0, 1))
            estimated_points = np.zeros((0, 1))

            # PREDICTION
                        
            x_k_prior = x_k_posterior

            P_k_prior = P_k_posterior + covariance_of_process_model

            sigma_x_points = compute_sigma_points(x_k_prior, P_k_prior, alpha, beta, lambda_)

        
            R = compute_R(x_k_prior[3], x_k_prior[4], x_k_prior[5])
            
            y_real = R @ U + x_k_prior[:3].reshape(-1, 1)
            y_real = y_real.reshape((3*U.shape[1],),order='F')

            for i in range(U.shape[1]):

                #previously_selected_points = np.vstack((previously_selected_points,U[:,i].reshape(-1,1)))

                y_k_prior = R @ U[:,i].reshape(-1, 1) + x_k_prior[:3].reshape(-1, 1)
                estimated_points = np.vstack([estimated_points,y_k_prior])
            
            #Compute the propagated sigma points sigma_y
            
            sigma_y_points = np.zeros((3*U.shape[1],13))


            for i in range(13):
                
                state_k = sigma_x_points[:,i]
                R_sigma = compute_R(state_k[3], state_k[4], state_k[5])
                t_sigma = np.array([state_k[0],state_k[1],state_k[2]]).reshape(-1,1)
                state_y = R_sigma @ U + t_sigma
                sigma_y_points[:, i] = state_y.reshape((3*U.shape[1],),order='F')
            
            # Compute Pxy Py
            Pxy = 0
            Py = 0
            
            y_mean = 0
            
            for i in range(13):
                y_mean = y_mean + w_mean[i] * sigma_y_points[:, i]
            
            
            for i in range(13): 
                Py = Py + Wc[i] * (sigma_y_points[:, i] - y_mean.reshape(-1,1)) @ (sigma_y_points[:, i] - y_mean.reshape(-1,1)).T
                Pxy = Pxy + Wc[i] * (sigma_x_points[:,i].reshape(-1,1) - x_k_prior.reshape(-1,1)) @ (sigma_y_points[:, i].reshape(-1,1) - y_mean.reshape(-1,1)).T

               
            K_k = Pxy @ np.linalg.pinv(Py)
        
            #find the closest point in Y to the estimated y

            y_estimated = R @ U + x_k_prior[:3].reshape(-1, 1)
            closest_points_matrix1 = find_closest_point(Y,y_estimated)
            closest_points_matrix1 = closest_points_matrix1.reshape((3*closest_points_matrix1.shape[1],1),order='F')
            y_estimated = y_estimated.reshape((3*y_estimated.shape[1],1),order='F')
            
            
            
            #CORRECTION
            x_k_posterior = x_k_prior.reshape(-1,1) + K_k @ (closest_points_matrix1 - y_estimated)
            x_k_posterior = x_k_posterior.reshape(6)
            P_k_posterior = P_k_prior - K_k @ Py @ K_k.T
            #print(P_k_posterior)

            R = compute_R(x_k_posterior[3], x_k_posterior[4], x_k_posterior[5])
            t = np.array([x_k_posterior[0],x_k_posterior[1],x_k_posterior[2]]).reshape(-1,1)

            U = R @ U + t
            
            fre = compute_fre(Y,U,R,t)
            print(fre)
        


        T = np.identity(4)

        T[0:3,0:3] = R
        T[0,3] = x_k_posterior[0]
        T[1,3] = x_k_posterior[1]
        T[2,3] = x_k_posterior[2]


        return T, R, t

#T, R, t = UKF(U_init,E)
#fre1 = compute_fre(E,U_init,R,t)

#print(fre1)
#print(fre2)

def PCA_registration(points_U,points_Y):
    
    N_U = points_U.shape[1]
    N_Y = points_Y.shape[1]
    
    # calculate centroids
    U_centroid = np.reshape(1/N_U * (np.sum(points_U, axis=1)), (3,1))
    Y_centroid = np.reshape(1/N_Y * (np.sum(points_Y, axis=1)), (3,1))

    cov_U = np.cov(points_U)
    cov_Y = np.cov(points_Y)

    U_pca_points_U,S_pca_points_U,V_T_pca_points_U = np.linalg.svd(cov_U)
    U_pca_points_Y,S_pca_points_Y,V_T_pca_points_Y = np.linalg.svd(cov_Y)
    R_pca = V_T_pca_points_Y.T @ V_T_pca_points_U

    
    #print("det",np.linalg.det(R_pca))
    t_pca = Y_centroid - R_pca @ U_centroid

    return R_pca,t_pca





R_pca,t_pca = PCA_registration(U_noisy,E)
res1 = compute_fre(E, U_noisy, R_pca, t_pca)

R_arun,t_arun,T_arun = arun(U_noisy,E)
res2 = compute_fre(E,U_noisy, R_arun, t_arun)

U_init = R_arun @ U_noisy + t_arun

# Plot the point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(E[0], E[1], E[2], c='blue', marker='o', label='Point Cloud')
ax.scatter(U_init[0], U_init[1], U_init[2], c='red', marker='o', label='Point Cloud')
plt.show()



print(res1)
print(res2)

#U_init = R @ U_noisy + t

T, R, t = UKF(U_init,E)




