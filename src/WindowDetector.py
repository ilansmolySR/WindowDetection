import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def project_ortho(x, C, N):
    """
    Calculate an orthogonal projection of the points on the standard
    Args:
        x: (n-1) x m dimensional matrix
        C: n dimensional vector which indicate the centroid of the standard
        N: n dimensional vector which indicate the normal vector of the standard
    Returns:
        m dimensional vector which indicate the last attribute value oforthogonal projection
    """
    Ck = C[0:-1]    # centroid for known parameters
    Nk = N[0:-1]    # normal for known parmeters
    Cu = C[-1]      # centroid for unknown parameter
    Nu = N[-1]      # normal for unknown parameter
    return np.dot(x-Ck, Nk) * -1.0 / Nu + Cu

def fit_plane_Yerror(points):
    """
    fit a plane to set of points in 3D

    Parameters
    ----------
    points : list of tuple
        each tuple represent a point in 3D (x,y,z)

    """
    tmp_a = []
    tmp_b = []
    for p in points:
        tmp_a.append([p[0], p[2], 1])
        tmp_b.append(p[1])
    B = np.matrix(tmp_b).T
    A = np.matrix(tmp_a)

    fit = (A.T * A).I * A.T * B
    errors = B - A * fit
    residual = np.linalg.norm(errors)

    return fit

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors: \n", errors)
    print("residual:", residual)


def plot(points,fit,scale=5):
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(0,scale)
    ax.set_ylim(0,scale)
    ax.set_zlim(0,scale)

    for p in points:
        ax.scatter(*p,)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                      np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]


    ax.plot_wireframe(X,Z,Y, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()





def fit_plane_SVD(points):
    """
    fit plane via solving Singular Value Decomposition.
    """
    # Find the average of points (centroid) along the columns
    X = np.array(points)
    C = np.average(X, axis=0)
    CX = X - C # delta

    U, S, V = np.linalg.svd(CX)     # Singular value decomposition
    N = V[-1] # smallest eigenvalues (singular values).
    N = N #/ np.linalg.norm(N)
    x0, y0, z0 = C
    a, b, c = N

    d = (a * x0 + b * y0 + c * z0)

    return -a/b, -c/b, d/b


if __name__=="__main__":
    p1 = 1, 2, 1
    p2 = 2, 2.1, 2
    p3 = 3, 1.9, 2
    p4 = 4, 2, 1
    points = [p1,p2,p3,p4]

    points = np.array([[-1.04194694, -1.17965867,  1.09517722],
                    [-0.39947906, -1.37104542,  1.36019265],
                    [-1.0634807 , -1.35020616,  0.46773962],
                    [-0.48640524, -1.64476106,  0.2726187 ],
                    [-0.05720509, -1.6791781 ,  0.76964551],
                    [-1.27522669, -1.10240358,  0.33761405],
                    [-0.61274031, -1.52709874, -0.09945502],
                    [-1.402693  , -0.86807757,  0.88866091],
                    [-0.72520241, -0.86800727,  1.69729388]])
    points = points - np.min(points,axis=0)
    points = 5*points/np.max(points, axis=0)
    fit = fit_plane_Yerror(points)
    print("Yerror",fit)
    plot(points,fit)

    fit = fit_plane_SVD(points)
    print("SVD",fit)
    plot(points,fit)i
