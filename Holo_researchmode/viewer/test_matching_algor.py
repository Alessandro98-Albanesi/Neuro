import itertools
import numpy as np
import open3d as o3d
from UKF import UKF
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise 

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

##print(match)
#print(result)
#print(match_R)
print(min_err)

C = match_R @ match + match_t





#def UKF_registration(measured_points,fixed_points):

'''
    x_k_posterior = np.array([0,0,0,0,0,0]).reshape((6, 1))
    P_k_posterior = np.identity(6)

    sigma_x = 0.01  # Example value, adjust as needed
    sigma_y = 0.01  # Example value, adjust as needed
    sigma_z = 0.01  # Example value, adjust as needed

    # Create covariance matrix
    covariance_matrix_noise = np.diag([sigma_x, sigma_y, sigma_z])
    
    variances_fixed_points = np.var(fixed_points, axis=1)
    
    
    covariance_of_process_model = np.diag([sigma_x,sigma_y,sigma_z,180/((np.sqrt(variances_fixed_points[1]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_y)))),180/((np.sqrt(variances_fixed_points[0]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_x)))),180/((np.sqrt(variances_fixed_points[0]/sigma_y)+(np.sqrt(variances_fixed_points[1]/sigma_x))))]) 
    
    previously_selected_points = np.zeros((0, 1))
   
    estimated_points = np.zeros((0, 1))
    
    x_i = covariance_of_process_model

    for i in range(measured_points.shape[1]):

        x_k_prior = x_k_posterior
        #print("x_k_prior",x_k_prior)
        P_k_prior = P_k_posterior + covariance_of_process_model
        #print(P_k_prior)
        #sigma_point_k = compute_sigma_points(x_k_prior, P_k_prior, alpha=1e-3, beta=2.0, kappa=0.0).reshape(6,1)

        previously_selected_points =  np.vstack([previously_selected_points, fixed_points[:,i].reshape(3, 1)])

        theta_x = x_k_prior[3,0]
        theta_y = x_k_prior[4,0]
        theta_z = x_k_prior[5,0]
        
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

        R = np.dot(Rz, np.dot(Ry, Rx))
        
        y_k_prior = R @ measured_points[:,i].reshape(-1, 1) + x_k_prior[:3,0].reshape(-1, 1)
        
        estimated_points = np.vstack([estimated_points,y_k_prior])
        
        #print(estimated_points)
        #print(previously_selected_points)

        result_array_y = previously_selected_points - estimated_points 
        print("res",result_array_y.shape)
        reuslt_array_x =  (x_k_prior + 0.01) - x_k_prior

        Pxy = np.outer(reuslt_array_x, result_array_y) 
        Pyy = np.outer(result_array_y,result_array_y)
        
        print(Pxy)
        print(Pyy.shape)

        K_k = np.dot(Pxy, np.linalg.pinv(Pyy))
       
        x_k_posterior = x_k_prior + K_k @ result_array_y
        
        P_k_posterior = P_k_prior - K_k @ Pyy @ K_k.T
    
    return x_k_posterior
'''

















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

        R = np.dot(Rz, np.dot(Ry, Rx))

        return R

def compute_sigma_points(x, P, alpha=0.1, beta=2.0, kappa=0.0):
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
    lambda_ = alpha 

    # Compute the sigma points
    sigma_points = np.zeros((n, 2 * n + 1))

    # Compute sigma points as columns of matrix X
    sqrt_P = np.linalg.cholesky((n + lambda_) * P)

    sigma_points[:, 0] = x  # Mean as the first sigma point

    for i in range(n):
        sigma_points[:, i + 1] = x + sqrt_P[:, i]
        sigma_points[:, n + i + 1] = x - sqrt_P[:, i]

    return sigma_points


def UKF(moving_points, fixed_points):

        x_k_posterior = np.array([0,0,0,0,0,0])
        P_k_posterior = np.identity(6)
        sigma_x = 0.01  # Example value, adjust as needed
        sigma_y = 0.01  # Example value, adjust as needed
        sigma_z = 0.01  # Example value, adjust as needed

        # Create covariance matrix
        covariance_matrix_noise = np.diag([sigma_x, sigma_y, sigma_z])

        variances_fixed_points = np.var(Y, axis=1)

        w_mean = np.zeros(13)
        lambda_ = 0.01 * 6 
        alpha=0.01 
        beta=0.15
        w_mean[0] = lambda_ / (lambda_ + 6) + (1 - alpha*alpha + beta)
        w_mean[1:12] = 1/ (2*(lambda_ + 6))

        covariance_of_process_model = np.diag([sigma_x,sigma_y,sigma_z,180/((np.sqrt(variances_fixed_points[1]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_y)))),180/((np.sqrt(variances_fixed_points[0]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_x)))),180/((np.sqrt(variances_fixed_points[0]/sigma_y)+(np.sqrt(variances_fixed_points[1]/sigma_x))))]) 


        sigma_points_cumulative = np.empty((13,0))

        previously_selected_points = np.zeros((0, 1))

        estimated_points = np.zeros((0, 1))

        for i in range(X.shape[1]):
                
            sigma_y_points = np.empty((0,3))
            x_k_prior = x_k_posterior
            P_k_prior = P_k_posterior + covariance_of_process_model
            sigma_x = compute_sigma_points(x_k_posterior, P_k_posterior)
            
            
            for k in range(sigma_x.shape[1]):

                R_k = compute_R(sigma_x[3][k], sigma_x[4][k], sigma_x[5][k])
                #print(sigma_x[3][k])
                y_sigma = R_k @ X[:,i] + sigma_x[0:3,k]
                sigma_y_points = np.vstack((sigma_y_points,y_sigma))

            sigma_points_cumulative = np.hstack((sigma_points_cumulative,sigma_y_points))
            sigma_points_cumulative_transpose = sigma_points_cumulative.T
            
            R_k = compute_R(x_k_prior[3],x_k_prior[4],x_k_prior[5])
            y_k = R_k @ moving_points[:,i] + x_k_prior[0:3]
            
            Pyy = 0
            Pxy = 0
            estimated_points = np.vstack((estimated_points,y_k.reshape(-1,1)))
            previously_selected_points = np.vstack((previously_selected_points,fixed_points[:,i].reshape(-1,1)))
            

            for k in range(13):
                
                Pyy = Pyy + w_mean[k]*(sigma_points_cumulative_transpose[:,k].reshape(-1,1) - estimated_points) @ (sigma_points_cumulative_transpose[:,k].reshape(-1,1) - estimated_points).T
                Pxy = Pxy + w_mean[k]*(sigma_x[:,k].reshape(-1,1) - x_k_posterior.reshape(-1,1)) @ (sigma_points_cumulative_transpose[:,k].reshape(-1,1) - estimated_points).T
            
            K_k = Pxy @ np.linalg.inv(Pyy)
            
            x_k_posterior = x_k_prior.reshape(-1,1) + K_k @ (previously_selected_points - estimated_points)
            x_k_posterior = x_k_posterior.reshape(6)

            P_k_posterior = P_k_prior - K_k @ Pyy @ K_k.T
       
        R = compute_R(x_k_posterior[3], x_k_posterior[4], x_k_posterior[5])
        t = np.array([x_k_posterior[0],x_k_posterior[1],x_k_posterior[2]]).reshape(-1,1)
        T = np.identity(4)

        T[0:3,0:3] = R
        T[0,3] = x_k_posterior[0]
        T[1,3] = x_k_posterior[1]
        T[2,3] = x_k_posterior[2]

       
        return T, R, t

U = points = np.random.uniform(-0.250, 0.250, size=(3, 30))

rotation_angles = np.random.uniform(-90, 90, size=(3,))


translation = np.random.uniform(-0.090, 0.090, size=(3,))


R_gener = compute_R(rotation_angles[0], rotation_angles[1], rotation_angles[2])
t_gener = translation


E = R_gener @ U + t_gener.reshape(-1,1)

covariance = 0.001 * np.eye(3)  # Identity matrix representing the covariance matrix
noise = np.random.multivariate_normal(mean=np.zeros(3), cov=covariance, size=U.shape[1]).T


U_noisy = U + noise


T, R, t = UKF(U_noisy, E)

S = R @ U_noisy + t.reshape(-1,1)


print(match_T)
# Calculate the squared Euclidean distance between corresponding points
squared_distance = np.sum((C - Y)**2, axis=0)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean(squared_distance))

print(rmse)
