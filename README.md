This project focuses on solving the image deblurring problem, modeling it as a convolution between a Point Spread Function (PSF) and the image. By reducing the problem to a mean square error minimization, we apply the conjugate gradient method to solve it.

The core of the algorithm involves multiplying a Toeplitz matrix with the image. By exploiting the properties of Toeplitz matrices, we use the Fast Fourier Transform (FFT) to perform this multiplication efficiently. This implementation integrates the Cooley-Tukey algorithm, Rader's algorithm, and cache optimization techniques for improved performance.

The code also includes a version that uses NumPy’s built-in FFT function, enabling the deblurring of larger images.
