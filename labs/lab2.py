import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def decompose(A):
    U = A.copy()
    L = np.zeros(A.shape)
    dim = len(L)
    for i in range(dim):
        L[i][i] = 1
    
    for i in range(dim):
        pivot = U[i][i]
        if i != dim - 1:
            for j in range(i+1, dim):
                L[j][i] = U[j][i] / pivot
                U[j] -= L[j][i] * U[i] 
    
    inverted_L = np.identity(dim)
    compliment_L = L.copy()

    inverted_U = np.identity(dim)
    compliment_U = U.copy()

    for i in range(dim):
        if i!= dim - 1:
            for j in range(i+1, dim):
                inverted_L[j] -= compliment_L[j][i] * inverted_L[i]

    for i in range(dim-1, -1, -1):
        inverted_U[i] *= 1/compliment_U[i][i] 
        if i!= 0: 
            for j in range(i-1, -1, -1):
                inverted_U[j] -= compliment_U[j][i] * inverted_U[i]



    return L, U, inverted_L, inverted_U, inverted_U @ inverted_L

A = (1/6) * np.array([[5.0,4.0,3.0, 2., 1.], [4.0,8.0,6.0, 4., 2.], [3.0,6.0,9.0, 6., 3.], [2., 4., 6., 8., 4.], [1., 2., 3., 4., 5.]])
#A = 1/6 * np.array([[5.0,4.0,3.0, 2.0, 1.0], [4.0,8.0,6.0, 4.0, 2.0], [3.0,6.0,9.0, 6.0, 3.0], [2.0, 4.0, 6.0, 8.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]])

L, U, inv_L, inv_U, inv_A = decompose(A)
b = [[0], [1], [2], [3], [4]]
#print(L)
#print(U)
#print(inv_L)
#print(inv_U)
res = inv_A @ A - np.identity(len(A))
#print(inv_A @ b)
print(np.mean(res))

def method_of_powers(A, max_iterations, min_diff):
    determinant = np.linalg.det(A)
    dim = len(A)
    i = 0
    while np.isclose(determinant, 0.0, 1e-5):
        A += np.identity(dim)
        determinant = np.linalg.det(A)
        i += 1
    print(f"new determinant is {determinant}")
    old_vect = np.array([[1] for i in range(dim)])
    new_vect = old_vect.copy() * 1000
    iteration = 0
    while (np.sum((np.abs(new_vect - old_vect))) > min_diff) and (iteration < max_iterations):
        old_vect = new_vect.copy()
        new_vect = A @ old_vect
        iteration += 1
        new_vect = new_vect/np.linalg.norm(new_vect)
    eigenvalue = (new_vect.T @ A) @ new_vect
    print(f"largest eigenvalue estimate is {eigenvalue - i} in {iteration} iterations")
    
    

A = np.array([[0., -0.5, 0., 0., 0.], [-0.5, 0., -0.5, 0., 0.], [0., -0.5, 0., -0.5, 0.], [0., 0., -0.5, 0., -0.5], [0., 0., 0., -0.5, 0.]])
#A*= 2/3
#method_of_powers(A, 10000, 1e-9)
def invert_diagonal_matrix(A):
    for i in range(len(A)):
        if A[i][i] != 0.:
            A[i][i] = 1/A[i][i]
        else: 
            raise ValueError("Non invertable matrix")
    return A
def check_for_diagonal_dominance(A):
    for i in range(len(A)):
        if np.sum(abs(A[i])) - 2 * abs(A[i][i]) >= 0:
            raise()
def jacobi_method(A, b, max_iterations, min_diff):
    dim = len(A)
    diagonal = np.zeros(shape= (dim, dim))
    for i in range(dim):
        if A[i][i] == 0:
            raise ValueError("Cannot have non zero diagonal elements for this matrix to be used in the Jacobi Method") 
        diagonal[i][i] = A[i][i]
    S = A - diagonal
    B_invert = invert_diagonal_matrix(diagonal)
    x = np.array([[0] for i in range(dim)])
    old_x = np.array([[1000] for i in range(dim)])
    iter = 0
    while (np.sum((np.abs(old_x - x))) > min_diff) and (iter < max_iterations):
        old_x = x.copy()
        x = B_invert @ (b-(S @ x))
        iter += 1
    print(f"Solutions are {x} in {iter} iterations")
    return x


A = np.array([[2., -1., 0., 0., 0.], [-1., 2., -1., 0., 0.], [0., -1., 2., -1., 0.], [0., 0., -1., 2., -1.], [0., 0., 0., -1., 2.]])
b = ([-1], [0], [0], [0], [5])

#x = jacobi_method(A, b, 10000, 1e-9)

r = np.array([[0], [0]])

q1 = {'charge': 0.5,
      'pos':  np.array([[-0.5], [np.sqrt(3) / 2]])}
q2 = {'charge': 1,
      'pos':  np.array([[-0.5], [-np.sqrt(3) / 2]])}
q0 = {'charge': 1,
      'pos':  np.array([[1.], [0.]])}

charges = [q0, q1, q2]


def electric_field(r, charges):
    field = np.array([[0.], [0.]])
    for charge in charges:
        field += charge['charge'] * ((r - charge['pos'])/((np.linalg.norm(r-charge['pos']))**3))
    return field
def generate_jacobian_inverse(r, charges, step_size, field_val):
    x, y = r[0][0], r[1][0]
    dxx = (electric_field(np.array([[x + step_size], [y]]), charges)[0, 0] - field_val[0, 0]) / step_size
    dxy = (electric_field(np.array([[x], [y + step_size]]), charges)[0, 0] - field_val[0, 0]) / step_size
    dyx = (electric_field(np.array([[x + step_size], [y]]), charges)[1, 0] - field_val[1, 0]) / step_size
    dyy = (electric_field(np.array([[x], [y + step_size]]), charges)[1, 0] - field_val[1, 0]) / step_size
    inverse_jacobian = (1/(dxx * dyy - dxy * dyx)) * np.array([[dyy, -dxy], [-dyx, dxx]])
    return inverse_jacobian

def find_equil(charges, init_guess, max_iterations, min_diff):
    pos = init_guess
    old_pos = init_guess + np.array([[1000], [1000]])
    iteration = 0
    while (((np.linalg.norm(pos - old_pos))) > min_diff) and (iteration < max_iterations):
        old_pos = pos.copy()
        step_size = 1e-10
        field_val = electric_field(pos, charges)
        #print(field_val)
        inv_jac = generate_jacobian_inverse(pos, charges, step_size, field_val)
        #print(inv_jac)
        pos -= inv_jac @ field_val
        iteration += 1
    print(f"position is {pos}, found in {iteration} iterations. E field at this point is {electric_field(pos, charges)}")

find_equil(charges, np.array([[0.], [0.]]), 1000, 1e-9)