import math
import random


#This file contains different matrix multiplication algorithms
def standard_multiplication(a, b):
    #input: two matrices
    #Output: the product of the two matrices
    #Takes in 2 matrices and returns the product of the two using standard algorithm
    # Dimensions
    rowsA, colsA = len(a), len(a[0])
    rowsB, colsB = len(b), len(b[0])

    # Check if multiplication is defined
    if colsA != rowsB:
        raise ValueError(
            f"Cannot multiply: a is {rowsA}×{colsA}, b is {rowsB}×{colsB}. "
            "Inner dimensions must match."
        )

    # Initialize result matrix with zeros (size: rowsA × colsB)
    c = [[0 for _ in range(colsB)] for _ in range(rowsA)]

    # Standard triple-loop multiplication
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for k in range(colsA):
                total += a[i][k] * b[k][j]
            c[i][j] = total

    return c

def add(A, B):
    """Matrix addition."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def subtract(A, B):
    """Matrix subtraction."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def split_matrix(M):
    """Split matrix into quarters."""
    n = len(M)
    mid = n // 2
    A11 = [row[:mid] for row in M[:mid]]
    A12 = [row[mid:] for row in M[:mid]]
    A21 = [row[:mid] for row in M[mid:]]
    A22 = [row[mid:] for row in M[mid:]]
    return A11, A12, A21, A22

def combine_quadrants(C11, C12, C21, C22):
    """Combine four submatrices into one."""
    top = [c11 + c12 for c11, c12 in zip(C11, C12)]
    bottom = [c21 + c22 for c21, c22 in zip(C21, C22)]
    return top + bottom

def pad_matrix(M, size):
    """Pad matrix with zeros to reach the desired size."""
    padded = [row + [0] * (size - len(row)) for row in M]
    padded += [[0] * size for _ in range(size - len(M))]
    return padded

def unpad_matrix(M, rows, cols):
    """Remove zero-padding."""
    return [row[:cols] for row in M[:rows]]

def strassen(A, B):
    """Strassen's matrix multiplication algorithm."""
    n = len(A)

    # Base case: 1x1 matrix
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # Split matrices
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # Compute the 7 products (recursively)
    M1 = strassen(add(A11, A22), add(B11, B22))
    M2 = strassen(add(A21, A22), B11)
    M3 = strassen(A11, subtract(B12, B22))
    M4 = strassen(A22, subtract(B21, B11))
    M5 = strassen(add(A11, A12), B22)
    M6 = strassen(subtract(A21, A11), add(B11, B12))
    M7 = strassen(subtract(A12, A22), add(B21, B22))

    # Compute result quadrants
    C11 = add(subtract(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(subtract(add(M1, M3), M2), M6)

    return combine_quadrants(C11, C12, C21, C22)

def strassen_multiply(A, B):
    """Wrapper to handle any NxM matrices by padding."""
    rowsA, colsA = len(A), len(A[0])
    rowsB, colsB = len(B), len(B[0])

    # Ensure matrices can be multiplied
    if colsA != rowsB:
        raise ValueError("Inner dimensions must match.")

    # Find next power of 2
    n = max(rowsA, colsA, rowsB, colsB)
    m = 1 << math.ceil(math.log2(n))

    # Pad
    A_padded = pad_matrix(A, m)
    B_padded = pad_matrix(B, m)

    # Multiply using Strassen
    C_padded = strassen(A_padded, B_padded)

    # Unpad result
    return unpad_matrix(C_padded, rowsA, colsB)

def random_matrix(n, m, value):
    """Generate random matrix of size n x m."""
    if value == 1:
        a = [[1 for j in range(m)] for i in range(n)]
    elif value == 2:
        a = [[2 for j in range(m)] for i in range(n)]
    elif value == 4:
        a = [[random.randint(1000, 9999) for j in range(m)] for i in range(n)]
    else:
        a = [[random.randint(1, 10) for j in range(m)] for i in range(n)]
    return a
