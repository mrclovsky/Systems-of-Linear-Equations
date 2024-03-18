import numpy as np
import matplotlib.pyplot as plt
import math
import time

def jacobi(A, b, res, max_iter=100):
    n = len(b)
    x = [0]*n
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        r = [b[i] - sum(A[i][j] * x_new[j] for j in range(n)) for i in range(n)]
        norm = sum(r[i] ** 2 for i in range(n)) ** 0.5
        if norm < res:
            return x_new, k
        x = x_new.copy()
    return x, max_iter

def gauss_seidel(A, b, res, max_iter=100):
    n = len(b)
    x = [0]*n
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        r = [b[i] - sum(A[i][j] * x_new[j] for j in range(n)) for i in range(n)]
        norm = sum(r[i] ** 2 for i in range(n)) ** 0.5
        if norm < res:
            return x_new, k
        x = x_new.copy()
    return x, max_iter

def LU_factorization(A, n):
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for i in range(n):
        for j in range(i, n):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - s1

    for j in range(i + 1, n):
        s2 = sum(U[k][i] * L[j][k] for k in range(i))
        L[j][i] = (A[j][i] - s2) / U[i][i]

    L[i][i] = 1.0
    return L, U

def LU_solve(b, n):
    y = [0.0] * n
    x = [0.0] * n

    # Ly = b for y
    for i in range(n):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = b[i] - s

    # Ux = y for x
    for i in range(n - 1, -1, -1):
        s = sum(U[i][k] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x, y

def LU_norm(A, b, n):
    r = [0.0] * n
    for i in range(n):
        r[i] = b[i] - sum(A[i][j] * x[j] for j in range(n))

    norm = math.sqrt(sum(r[i] ** 2 for i in range(n)))

    return norm

if __name__ == '__main__':

##A
    e = 7
    c = 2
    d = 4
    f = 8

    a1 = 5 + e
    a2 = -1
    a3 = -1

    N = 924
    b = [0]*N

    for n in range(1, N + 1):
        b[n - 1] = math.sin(n * (f + 1))

    A = np.zeros((N,N))

    for i in range(N):
        A[i][i] = a1
    for i in range(N - 1):
        A[i + 1][i] = a2
        A[i][i + 1] = a2
    for i in range(N - 2):
        A[i + 2][i] = a3
        A[i][i + 2] = a3

    print(b)
    print(A)
    print()

##B
    res = 1e-9

    start = time.time()
    x, k = jacobi(A, b, res)
    end = time.time()
    print("Jacobi method:")
    print("Time: " + str(end-start) + " Number of iterations: " + str(k))
    print()

    start = time.time()
    x, k = gauss_seidel(A, b, res)
    end = time.time()
    print("Gauss-Seidel method:")
    print("Time: " + str(end-start) + " Number of iterations: " + str(k))
    print()

##C
    a1 = 3
    for i in range(N):
        A[i][i] = a1

    print(A)
    print()

    start = time.time()
    x, k = jacobi(A, b, res)
    end = time.time()
    print("Jacobi method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

    start = time.time()
    x, k = gauss_seidel(A, b, res)
    end = time.time()
    print("Gauss-Seidel method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

##D
    L, U = LU_factorization(A, N)
    x, y = LU_solve(b, N)
    norm = LU_norm(A, b, N)

    print("Direct method:")
    print("Residual norm:", norm)
    print()

##E
    a1 = 5 + e
    N = [100, 500, 1000, 2000, 3000]
    jacobi_times = [0]*len(N)
    gauss_times = [0]*len(N)
    direct_times = [0]*len(N)

    for k in range(len(N)):
        b = [0] * N[k]

        for n in range(1, N[k] + 1):
            b[n - 1] = math.sin(n * (f + 1))

        A = np.zeros((N[k], N[k]))

        for i in range(N[k]):
            A[i][i] = a1
        for i in range(N[k] - 1):
            A[i + 1][i] = a2
            A[i][i + 1] = a2
        for i in range(N[k] - 2):
            A[i + 2][i] = a3
            A[i][i + 2] = a3

        start = time.time()
        x, itr = jacobi(A, b, res)
        end = time.time()
        jacobi_times[k] = end - start

        start = time.time()
        x, itr = gauss_seidel(A, b, res)
        end = time.time()
        gauss_times[k] = end - start

        start = time.time()
        L, U = LU_factorization(A, N[k])
        x, y = LU_solve(b, N[k])
        norm = LU_norm(A, b, N[k])
        end = time.time()
        direct_times[k] = end - start

    print(jacobi_times)
    print(gauss_times)
    print(direct_times)

    plt.plot(jacobi_times, N, label='Jacobi')
    plt.plot(gauss_times, N, label='Gauss-Seidel')
    plt.plot(direct_times, N, label='Direct')

    plt.xlabel('Time [s]')
    plt.ylabel('N')
    plt.title('Algorithm Execution Times')

    plt.legend()
    plt.savefig("AlgorithmExecutionTimes.png")
    plt.show()