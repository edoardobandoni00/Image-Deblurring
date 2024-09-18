# Image-Deblurring

This project focuses on solving the image deblurring problem, modelling it as a convolution between a Point Spread Function (PSF) and the image. By reducing the problem to a mean square error minimisation, the conjugate gradient method is applied to solve it.

The core of the algorithm involves multiplying a Toeplitz matrix by the image. By exploiting the properties of Toeplitz matrices, the Fast Fourier Transform (FFT) is used to perform this multiplication efficiently. This implementation integrates the Cooley-Tukey algorithm, Rader's algorithm, and cache optimisation techniques for an improved performance.

The project also includes a version that uses NumPy’s built-in FFT function, enabling the deblurring of larger images.
