# Linear Algebra


## Contents  
I - Linear Algebra  
II - Functions  
III - Performance  
  
# I - Linear Algebra

Contains basic algorithlm for linear algebra and is numpy compatible.

# II - Functions

Here is the list of implemented functions:

- **is_sym(A)**  
  Tests whether the matrix `A` is symmetric.  
  Complexity: **O(n³)**

- **determinant_naive(A)**  
  Computes the determinant of `A` using the basic recursive formula.  
  Complexity: **O(n!)**

- **determinant_gauss(A)**  
  Computes the determinant of `A` using Gaussian elimination.  
  Complexity: **O(n³)**

- **is_def_pos_sylvester(A)**  
  Tests whether the symmetric matrix `A` is positive definite using Sylvester’s criterion.

- **inverse_gauss(A)**  
  Computes the inverse of `A`, if it exists, using Gaussian elimination.

- **cholesky(A)**  
  Computes the Cholesky decomposition `L` such that  
  `A = L * L.T` for a symmetric positive definite matrix `A`.

- **eigen(A)**  
  Computes the eigenvalues and eigenvectors of `A` if the matrix is diagonalizable.

# III - Performance

- The determinant of a `(1000, 1000)` matrix is computed in approximately **2 seconds** using Gaussian elimination.
- The diagonalization of a `(500, 500)` matrix is performed in approximately **10 seconds**.