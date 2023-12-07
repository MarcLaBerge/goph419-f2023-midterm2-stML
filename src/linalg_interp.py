import numpy as np


NONE = 0
MAX_ITERATIONS = 100


def gauss_iter_solve(A, b, x0=NONE, tol= 1e-8, alg= 'seidel'):
    """
    Using iterative Gauss-Seidel to solve a linear system
    
    Parameters
    ----------
    A:
        An array containing the coefficient matrix
    b:
        An array containing the right-hand-side vector(s)
    x0: 
        An array containing initial guess(es)
        Defult guess is 0 or None
        When a value is entered it should be the same shape as b
        Raise error if A,b,x0 shapes arn't compatable
    tol: 
        A float of felative error tolerance or stopping criterial
        Defult parameter = 1e-8
        Raise RuntimeWarning if the solution doesn't converge after a specified number of iterations
    alg: 
        A string flag for the algorithm to be used
        Option for 'seidel' or 'jacobi' iteration algorithm should be used
        Raise error if neither option above is entered
    """
    #Default
    x0 = NONE
    tol = 1e-8
    alg = 'seidel'


    #Raising Errors
        #array like check
    A = np.array(A, dtype = float)
    b = np.array(b, dtype = float)

    #Checking if A is 2D
    m = len(A) #returns number of rows
    ndim = len(A.shape)
    if ndim != 2: #has to be a 2d array
        raise ValueError(f"A has {ndim} dimensions" + ", should be 2")

    #Checking that A is square
    if A.shape[1] != m: #Checking that the amounto of columns in row 1 is the same as the amount of rows
        raise ValueError(f"A has {m} rows and {A.shape[1]} columns" + ", should be square")
    

    #Checking if b is 1D 
    ndimb = len(b.shape)
    if ndimb != 1 and ndimb != 2:
        raise ValueError(f"b has {ndimb} dimensions" + ", should be 1D or 2D")
    
    #Checking that A and b are compatable
    n = len(b) #amount of rows
    if n != m:
        raise ValueError(f"A has {m} rows, B has {n} values" + ", dimensions incompatible")
    

    # Initialize an inital guess of zeros if no intial guess provided and check that x and b have the same shape
    if not x0:
        x = np.zeros_like(b)
    else:
        x = np.array(x0,dtype=float)
        if x.shape != b.shape:
            raise ValueError(f"X has shape {x.shape}, b has shape {b.shape}"+ ", should be same length") 
    
    # Check that alg is one of 'seidel' or 'jacobi'

    if alg.strip().lower() not in ('seidel','jacobi'):
        raise ValueError("Unrecognized iteration algorithm"+ ", choose either 'seidel' or 'jacobi'")
    
    #######################################################################################################################

    # Initialize error and iteration counter
    eps_a = 2 * tol
    count = 0
    

    # Normalize coefficient matrix and b vector
    A_diag = np.diag(1.0/np.diag(A))
    b_star = A_diag @ b
    A_star = A_diag @ A
    A_s = A_star - np.eye(m)

    # Perform Gauss-Seidel iteration based on alg first jacobi

    if alg.strip().lower() == 'jacobi':

        # Perform Gauss-Seidel method until convergence or maximum iterations

        while count < MAX_ITERATIONS and eps_a > tol:
            # Replace old x with new x
            xo = x.copy() 
            count += 1
            x = b_star - A_s @ x 
            # New vs old guess
            dx = x - xo 
            # Check for any errors
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x) 

    else:

        # Perform Jacobi method until convergence or maximum iterations

        while eps_a > tol and count < MAX_ITERATIONS:

            xo = x.copy()
            count += 1 
            # Update each element in the x vector
            for i,a_row in enumerate(A_s):
                x[i] = b_star[i] - np.dot(a_row,x)
            # New vs old guess
            dx = x - xo 
            # Check for any erros
            eps_a = np.linalg.norm(dx) / np.linalg.norm(x) 

    # Check max interations vs counter
    if count >= MAX_ITERATIONS:
        raise RuntimeWarning(f"No convergence after {MAX_ITERATIONS} iterations"+ ", returning last updated x vector")
    
    return x

def spline_function(xd,yd,order=3):

    """ n order functions for spline interpolation.

    Parameters
    ----------
    xd : 
        numpy.ndarray, 
        shape=(n,1)
        vector of unique x values in ascending order
    yd : 
        numpy.ndarray, 
        shape=(n,1)
        vector of y values
    order : 
        int, 1 or 2 or 3
        order of interpolating function

    Returns
    -------
    s{order} : 
        function
        Interpolating function
    """
    # Default 
    order = 3

    # Check that xd is sorted in ascending order and reorder if necessary
    k_sort = np.argsort(xd) # Array of coefficients for sorted xd vector
    xd = np.array([xd[k] for k in k_sort]) # Sort xd
    yd = np.array([yd[k] for k in k_sort]) # Sort yd

    # Check that xd and yd have same shape
    if (xd.shape != yd.shape):
        raise ValueError(f"xd has length {len(xd)}, yd has length {len(yd)}" + ", should be same")
    
    # Check that order is 1 or 2 or 3
    if order not in (1,2,3):
        raise ValueError(f"Chosen order of {order} not supported.")
    
    # Initialize first order differences, f1 vector, a vector
    
    N = len(xd) # Number of data points
    a = yd[:-1] # Trimmed yd vector
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference

    # Calculate interpolation function based on order
    if order == 1:

        # Map first order differences to b
        b = f1

        # Define first order function
        def s1(x):

            # Assign spline location based on value in x
            k = (0 if x <=  xd[0]
                else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            # Return interpolated value at x
            return a[k] + b[k] * (x - xd[k])
        
        # Return first order function
        return s1
    
    elif order == 2:

        # Second order coefficient matrix
        
        A0 = np.hstack([np.diag(dx[:-1]),np.zeros((N-2,1))]) # Stack 0 vector to left edge of right-side lower portion
        A1 = np.hstack([np.zeros((N-2,1)),np.diag(dx[1:])]) # Stack 0 vector to right edge of left-side lower portion
        A = np.vstack([np.zeros((1,N-1)),(A0+A1)]) # Stack 0 row above combined A0 and A1 matrices
        B = np.zeros((N-1,)) # Initialize b vector
        B[1:] = np.diff(f1) # Second order differences
        A[0,:2] = [1,-1] # Add constant entries to first row

        # Solve for c coefficients w/ numpy.linalg (not sufficiently diagonally dominant for Gauss Seidel) and compute b vector
        c = np.linalg.solve(A,B)
        b = f1 - c * dx

        # Define second order spline function
        def s2(x):

            # Assign spline location based on value in x

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            # Return interpolated values at x
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2
        
        # Return second order function
        return s2
    
    else:

        # Third order coefficient matrix
        
        A = np.zeros((N,N)) # Initialize N by N 0 matrix
        A[1:-1,:-2] += np.diag(dx[:-1]) # Add lower off-diagonal entries, exclude first and last rows
        A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:])) # Add main diagonal entries, exclude first and last rows
        A[1:-1,2:] += np.diag(dx[1:]) # Add upper off-diagonal entries, exclude first and last rows
        A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]] # Add first row entries
        A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]] # Add last row entries

        # Design B vector

        B = np.zeros((N,))
        B[1:-1] = 3 * np.diff(f1)

        # Use gauss_iter_solve() function to compute c coefficients
        c = gauss_iter_solve(A,B)

        # Compute d coefficients
        d = np.diff(c) / (3 * dx)

        # Third order difference
        b = f1 - c[:-1] * dx - d * dx ** 2

        # Define third order spline function
        def s3(x):
            # Assign spline location based on value in x
            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
        
            # Return interpolated values at x
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3
        
        # Return third order function
        return s3
    
def build(xd,yd):

    N = len(xd) # Number of data points
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference

    A = np.zeros((N,N)) # Initialize N by N 0 matrix
    A[1:-1,:-2] += np.diag(dx[:-1]) # Add lower off-diagonal entries, exclude first and last rows
    A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:])) # Add main diagonal entries, exclude first and last rows
    A[1:-1,2:] += np.diag(dx[1:]) # Add upper off-diagonal entries, exclude first and last rows
    A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]] # Add first row entries
    A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]] # Add last row entries

    # Design B vector
    B = np.zeros((N,))
    B[1:-1] = 3 * np.diff(f1)

    return A,B

def check_dom(A):

    for i in range(len(A)):

        on_d = abs(A[i,i])
        off_d = 0
        
        for j in range(len(A)):

            off_d += abs(A[i,j])

        off_d -= on_d

        if off_d > on_d:

            return False

    return True

def norm_sys(A,B):

    m = len(B)
    a_diag = np.diag(1.0/np.diag(A))
    b_star = a_diag @ B
    a_star = a_diag @ A
    a_s = a_star - np.eye(m)

    return a_s,b_star

def jacobi(a,b):

    count = 0
    tol = 1e-8
    x = np.zeros(np.shape(b))
    MAX_ITERNATIONS = 100
    eps_a = 2 * tol

    while eps_a > tol and count < MAX_ITERATIONS:

        xo = x.copy() # Copy new x as old x
        count += 1 # Increase iteration counter
        x = b - a @ x # Compute new x guess
        dx = x - xo # Calculate difference between old and new guess
        eps_a = np.linalg.norm(dx) / np.linalg.norm(x) # Relative error update

    if count >= MAX_ITERATIONS:

        print(f"No convergence after {MAX_ITERATIONS} iterations. Returning last guess.")

    else:

        print("System solved!")

    return(x)

def solve_spline_coefs(xd,yd,c):

    a = yd[:-1] # Trimmed yd vector
    dx = np.diff(xd) # Vector of xd differences
    dy = np.diff(yd) # Vector of yd differences
    f1 = dy/dx # First order difference
    d = np.diff(c) / (3 * dx)
    b = f1 - c[:-1] * dx - d * dx ** 2

    return a,b,d

def gen_spline_func(a,b,c,d,x,xd):

    def spline_func(x):

        # Assign spline location based on value in x

        k = (0 if x <=  xd[0]
                else len(a) - 1 if x >= xd[-1]
            else np.nonzero(xd<x)[0][-1])
        
        # Return interpolated values at x
        
        return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3
    
    # Return third order function

    return spline_func