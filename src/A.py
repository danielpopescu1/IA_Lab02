import numpy as np

np.random.seed(42)

A = np.random.randint(1, 11, (4, 3))
B = np.random.randint(1, 11, (3, 5))

print("Matrix A (4x3):")
print(A)
print("\nMatrix B (3x5):")
print(B)

C = A @ B
print("\nMatrix product C = A @ B (4x5):")
print(C)

sum_all = np.sum(C)
mean_columns = np.mean(C, axis=0)
global_max = np.max(C)

print(f"\nSum of all elements in C: {sum_all}")
print(f"Mean of each column in C: {mean_columns}")
print(f"Global maximum in C: {global_max}")

M = np.random.randint(1, 11, (3, 3))
print("\nBonus - Matrix M (3x3):")
print(M)

M_inv = np.linalg.inv(M)
print("\nInverse of M:")
print(M_inv)

det_M = np.linalg.det(M)
print(f"\nDeterminant of M: {det_M}")

identity_check = np.allclose(M @ M_inv, np.eye(3))
print(f"\nM @ M_inv is close to identity matrix: {identity_check}")

print("\nM @ M_inv:")
print(M @ M_inv)
