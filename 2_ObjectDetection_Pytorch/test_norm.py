import numpy as np

X = np.array([4, 3])

# Norm: Chuan hoa de tinh ra mot vector co do dai bang 1
# L0Norm: so luong phan tu khac 0 trong vector
# L0Norm=2
l0norm = np.linalg.norm(X, ord=0)
print("L0Norm:", l0norm)

# L1Norm: Tong gia tri tuyet doi cua cac phan tu trong vector
# Khoang cach Manhattan
l1norm = np.linalg.norm(X, ord=1)
print("L1Norm:", l1norm)

# L2Norm: Do dai cua vector, Khoang cach Euclidean
# L2Norm = sqrt(4^2 + 3^2) = sqrt(25) = 5
l2norm = np.linalg.norm(X, ord=2)
print("L2Norm:", l2norm)