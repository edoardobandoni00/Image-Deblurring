from PIL import Image
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import toeplitz
import time


#### FFT Implementation
def next_factor(n):
    """Return the smallest factor of n greater than 1 using precomputed small primes."""
    if n <= 1:
        raise ValueError("Input must be greater than 1")
    
    # Small prime numbers for initial checking
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    # Check against small primes less than max_factor
    max_factor = int(np.sqrt(n))
    
    # First check against precomputed small primes
    for prime in small_primes:
        if prime > max_factor:
            return n, 1
        if n % prime == 0:
            return prime, n // prime
    
    for prime in range(47, max_factor + 1, 2):
        if n % prime == 0:
            return prime, n // prime
    
    # If no factor is found, return n (which means n is prime)
    return n, 1


def is_prime(n):
    """Return the smallest factor of n greater than 1 using precomputed small primes."""
    if n <= 1:
        raise ValueError("Input must be greater than 1")
    
    # Small prime numbers for initial checking
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

    # Check against small primes less than max_factor
    max_factor = int(np.sqrt(n))
    
    # First check against precomputed small primes
    for prime in small_primes:
        if prime > max_factor:
            return True
        if n % prime == 0:
            return False
    
    for prime in range(47, max_factor + 1, 2):
        if n % prime == 0:
            return False
    
    # If no factor is found n is prime
    return True


def prime_factors(n):
    """Finds prime factors of a given number n."""
    factors = set()
    
    # Small prime numbers for initial checking
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    
    # Calculate max factor
    max_factor = int(np.sqrt(n)) + 1
    
    # First check against precomputed small primes
    for prime in small_primes:
        if prime > max_factor:
            factors.add(n)
            n = 1
        while n % prime == 0:
            factors.add(prime)
            n //= prime
        if n == 1:
            return factors
    
    # Continue checking for factors beyond small primes
    for i in range(47, max_factor, 2):
        while n % i == 0:
            factors.add(i)
            n //= i
        if n == 1:
            return factors
    
    # If n is still greater than 1, it is prime and should be added to factors
    if n > 1:
        factors.add(n)
    
    return factors


def find_primitive_root(p):
    """Finds a primitive root modulo p."""
    if p == 2:
        return 1
    phi = p - 1
    factors = prime_factors(phi)
    for g in range(2, p):
        if all(pow(g, phi // factor, p) != 1 for factor in factors):
            return g
    raise ValueError(f"No primitive root found for {p}")


def cyclic_convolution(a, b):
    """Performs cyclic convolution using FFT."""
    A = fft1d(a)
    B = fft1d(b)
    C = A * B
    return ifft1d(C)


def apply_permutation(vector, permutation):
    """Applies a permutation to the last six indices of a vector."""
    
    permuted_vector = vector.copy()
    
    for i in range(1,len(permutation)):
       
        permuted_vector[permutation[i]] = vector[permutation[i-1]]
    
    permuted_vector[permutation[0]] = vector[permutation[-1]]
    
    return permuted_vector


def dft(x):
    """Computes the Discrete Fourier Transform (DFT) of a vector x using NumPy and precomputed twiddle factors."""
    N = len(x)
    x = np.asarray(x, dtype=complex)
    
    # Precompute the twiddle factors
    n = np.arange(N)
    k = n.reshape((N, 1))
    twiddle_factors = np.exp(-2j * np.pi * k * n / N)
    
    # Perform the matrix multiplication to get the DFT
    X = np.dot(twiddle_factors, x)
    
    return X


def rader_fft(x):
    """Computes the DFT of x using the special algorithm for prime lengths."""
    N = len(x)
    assert is_prime(N), "Length of the input sequence must be a prime number."

    # Find a primitive root modulo N
    g = find_primitive_root(N)

    # Compute X_0
    X_0 = np.sum(x)

    # Compute a_q and b_q
    a = np.zeros(N-1, dtype=complex)
    b = np.zeros(N-1, dtype=complex)

    for q in range(N-1):
        a[q] = x[pow(g, q, N)]
        b[q] = np.exp(-2j * np.pi * pow(g, -q, N) / N)

    # Perform cyclic convolution using FFT
    conv_result = cyclic_convolution(a, b)

    # Prepare the DFT result
    X = np.zeros(N, dtype=complex)
    X[0] = X_0

    for p in range(1, N):
        X[pow(g, -p, N)] = x[0] + conv_result[p-1]
        
    indeces = np.zeros(N-1,dtype=int)
        
    for p in range(N-1):
        indeces[p] = pow(g, p, N)
    
    return apply_permutation(X, indeces)


def fft1d(x):
    """Mixed-radix Cooley-Tukey FFT with optimizations"""
    N = len(x)
    if N <= 1:
        return x
    
    # If N is "small", compute the naive DFT to optimize cache space
    if N <= 60:
        return dft(x)
    
    if is_prime(N):
        return rader_fft(x)

    # If N is not prime, find the smallest factor to split the array
    p, q = next_factor(N)
    
    # FFT of subarrays
    X_p = [fft1d(x[k::p]) for k in range(p)]
    
    # Precompute twiddle factors
    twiddle_factors = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # Combine the results using in-place computation
    X = np.zeros(N, dtype=complex) 
    for k in range(N):
        for j in range(p):
            X[k] += X_p[j][k % q] * twiddle_factors[j * k % N]
    
    return X


def ifft1d(X): return np.conj(fft1d(np.conj(X)))/len(X)


def fft2d(matrix):
    # Compute the 2D FFT using row-column decomposition
    # Apply 1D FFT to each row
    rows = np.array([fft1d(row) for row in matrix])
    
    # Apply 1D FFT to each column of the row-transformed matrix
    cols = np.array([fft1d(rows[:, j]) for j in range(rows.shape[1])]).T
    
    return cols


def ifft2d(matrix):
    # Compute the 2D IFFT using row-column decomposition
    # Apply 1D IFFT to each row
    rows = np.array([ifft1d(row) for row in matrix])
    
    # Apply 1D IFFT to each column of the row-transformed matrix
    cols = np.array([ifft1d(rows[:, j]) for j in range(rows.shape[1])]).T
    
    return cols


#### Image Deblurring

def psf_sinc(k, alfa):
    """
    Create an exponential Point Spread Function (PSF) of size (2k+1)x(2k+1)
    with elements sinc(alfa*(i^2+j^2)^0.5), where i, j = -k, k.
    
    Parameters:
        k (int): Half the size of the PSF in each dimension.
        alfa (float): Parameter for adjusting the sinc function.
    
    Returns:
        numpy.ndarray: The created PSF.
    """
    x = np.arange(-k, k + 1).reshape(-1, 1) * np.ones((1, 2 * k + 1))
    y = x.transpose()
    psf = np.sinc(alfa * np.sqrt(x**2 + y**2))
    m = np.min(psf)
    psf = psf - m
    psf = psf / np.sum(psf)
    return np.double(psf)


def blur(a, psf, k):
    """
    Deconvolves an image using the given Point Spread Function (PSF) via FFT.
    """
    s = a.shape

    # Pad the PSF to the same size as the input image
    psf_padded = np.zeros((s[0],s[1]))
    psf_shape = psf.shape
    psf_padded[:psf_shape[0], :psf_shape[1]] = psf

    # Compute the FFT of the padded PSF
    psf_fft = np.fft.fft2(psf_padded)

    if len(s) == 3:
        b = np.zeros((s[0]-2*k, s[1]-2*k, 3), dtype=np.uint8)
        for i in range(s[2]):
            # FFT of the image channel
            a_fft = np.fft.fft2(a[:, :, i])
            
            # Convolution in frequency domain
            conv_result = np.fft.ifft2(a_fft * psf_fft)
            
            # Crop the result to the valid region 
            b[:, :, i] = np.real(conv_result[k:s[0]-k, k:s[1]-k])
            
    else:
        a_fft = np.fft.fft2(a)
        conv_result = np.fft.ifft2(a_fft * np.conj(psf_fft))
        b = np.real(conv_result[:s[0]-2*k, :s[1]-2*k])

    return np.clip(b,0,255).astype(np.uint8)


def my_fftpsf(psf, m, n):
    """
    Pad the PSF to the same size of a restored image. 
    Calculate the FFT of this zero-padded PSF matrix.
    """
    s = psf.shape
    a = np.zeros((m + s[0] - 1, n + s[1] - 1), dtype=psf.dtype)
    a[:s[0], :s[1]] = psf
    fpsf = fft2d(a)
    return fpsf


def my_fx(fpsf, x):
    """
    Compute the product y=Fx of the block-Toeplitz matrix F and a vector x.
    """
    M, N = fpsf.shape
    sx = x.shape
    xx = np.zeros((M, N), dtype=x.dtype)
    xx[:sx[0], :sx[1]] = x
    fx = fft2d(xx)
    y = ifft2d(fx * fpsf)
    return y


def my_ftx(fpsf, y, k):
    """
    Compute the product z = F^Ty of the transpose of the block-Toeplitz matrix F^T
    and a vector y.
    """
    w = fft2d(y)
    u = ifft2d(w * np.conj(fpsf))
    z = u[:-(2*k), :-(2*k)]  # Extracting the desired region
    return z


def my_gc_reg_toeplitz(b, psf, steps, reg, epsilon):
    times = np.zeros(steps)
    # Define the point-spread function (psf)
    psft = psf[::-1, ::-1]
    z = psft.shape
    k = int((z[0] - 1)/2) # k is an integer
    
    # Define the dimensions of the objects
    s = b.shape
    
    # Initialize the iterations of GC
    if len(s) == 3:
        x = np.zeros((s[0]+2*k,s[1]+2*k,s[2]))
        coln = 3
    else:
        x = np.zeros((s[0]+2*k,s[1]+2*k))
        coln = 1
    
    # psf associated to F^T
    psft = my_fftpsf(psf,s[0],s[1])
    for col in range(coln):
        if coln == 3:
            r = b[:, :, col]
        else:
            r = b
        
        r = np.double(r)
        y = np.zeros((s[0], s[1]))
        for iter in range(steps):
            start_time = time.time()
            rho = np.sum(np.sum(r * r))
            if iter == 0:
                p = r
            else:
                beta = rho / rhop
                p = r + beta * p
            
            tmp = my_fx(psft,p)
            q = my_ftx(psft,tmp,k)
            q = q + reg * p
            alpha = rho / np.sum(np.sum(p * q))
            y = y + alpha * p
            r = r - alpha * q
            rhop = rho
            nor = np.abs(np.sqrt(np.sum(np.sum(r * r))) / (s[0] * s[1]))
            end_time = time.time()
            elapsed_time = end_time - start_time
            times[iter] = elapsed_time
            print("Elapsed time:", elapsed_time, "seconds")
            print(f'Canale {col+1}, step={iter+1}, residuo={nor}')
            
            # Check convergence
            if nor < epsilon:
                print(f'Converged at step {iter+1} with residual {nor}')
                break
        
        if coln == 3:
            x[:, :, col] = np.real(my_fx(psft,y))
        else:
            x = np.real(my_fx(psft,y))
        
    x = np.clip(x,0,255).astype(np.uint8)
    print("Mean time per cycle:", np.mean(times), "seconds")
    
    return x


## TOEPLITZ
def fftpsf(psf, m, n):
    """
    Calculate the FFT of the first column of the block-circulant matrix
    associated with the PSF.

    Parameters:
        psf (numpy.ndarray): The PSF matrix.
        m (int): The number of rows in the resulting block-circulant matrix.
        n (int): The number of columns in the resulting block-circulant matrix.

    Returns:
        numpy.ndarray: The FFT of the first column of the block-circulant matrix.
    """
    s = psf.shape
    a = np.zeros((m + s[0] - 1, n + s[1] - 1), dtype=psf.dtype)
    a[:s[0], :s[1]] = psf
    fpsf = np.fft.fft2(a)
    return fpsf


def fx(fpsf, x):
    """
    Compute the product y=Fx of the block-Toeplitz matrix F and a vector x.

    Parameters:
        fpsf (numpy.ndarray): The FFT of the first column of the block-Toeplitz matrix F.
        x (numpy.ndarray): The input vector.

    Returns:
        numpy.ndarray: The product y=Fx.
    """
    m, n = fpsf.shape
    sx = x.shape
    xx = np.zeros((m, n), dtype=x.dtype)
    xx[:sx[0], :sx[1]] = x
    fx = np.fft.fft2(xx)
    y = np.fft.ifft2(fx * fpsf)
    return y


def ftx(fpsf, w, k):
    """
    Compute the product z = F^Tw of the transpose of the block-Toeplitz matrix F^T
    and a vector w.

    Parameters:
        fpsf (numpy.ndarray): The FFT of the first column of the block-Toeplitz matrix F.
        w (numpy.ndarray): The input vector.
        k (int): The size of the PSF kernel.

    Returns:
        numpy.ndarray: The product z = F^Tw.
    """
    fw = np.fft.fft2(w)
    u = np.fft.ifft2(fw * np.conj(fpsf))
    z = u[:-(2*k), :-(2*k)]  # Extracting the desired region
    return z


def gc_reg_toeplitz(b, psf, steps, reg, epsilon):
    times = np.zeros(steps)
    # Define the point-spread function (psf)
    psft = psf[::-1, ::-1]
    z = psft.shape
    k = int((z[0] - 1)/2) # k is an integer    
    
    # Define the dimensions of the objects
    s = b.shape
    
    # Initialize the iterations of GC
    if len(s) == 3:
        x = np.zeros((s[0]+2*k,s[1]+2*k,s[2]))
        coln = 3
    else:
        x = np.zeros((s[0]+2*k,s[1]+2*k))
        coln = 1
    
    # psf associated to F^T
    psft = fftpsf(psf,s[0],s[1])
    for col in range(coln):
        if coln == 3:
            r = b[:, :, col]
        else:
            r = b
        
        r = np.double(r)
        y = np.zeros((s[0], s[1]))
        for iter in range(steps):
            start_time = time.time()
            rho = np.sum(np.sum(r * r))
            if iter == 0:
                p = r
            else:
                beta = rho / rhop
                p = r + beta * p
            
            tmp = fx(psft,p)
            q = ftx(psft,tmp,k)
            q = q + reg * p
            alpha = rho / np.sum(np.sum(p * q))
            y = y + alpha * p
            r = r - alpha * q
            rhop = rho
            nor = np.abs(np.sqrt(np.sum(np.sum(r * r))) / (s[0] * s[1]))
            end_time = time.time()
            elapsed_time = end_time - start_time
            times[iter] = elapsed_time
            print("Elapsed time:", elapsed_time, "seconds")
            print(f'Channel {col+1}, step={iter+1}, residual={nor}')
        
            # Check convergence
            if nor < epsilon:
                print(f'Converged at step {iter+1} with residual {nor}')
                break
        
        if coln == 3:
            x[:, :, col] = np.real(fx(psft,y))
        else:
            x = np.real(fx(psft,y))
        
    x = np.clip(x,0,255).astype(np.uint8)
    print("Mean time per cycle:", np.mean(times), "seconds")
    
    return x



# Set parameters
k = 4
alpha = 0.4
psf = psf_sinc(k,alpha)
reg = 0.001
tolerance = 0

# TOY EXAMPLE

# Sharp image
A = np.array(Image.open('Desktop/Magistrale/Python/example.png'))

a = A[:,362:413,0]
img = Image.fromarray(A)
img.show()
plt.imshow(a, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = a.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(a[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.show()

# Blurred image
b = blur(A[:,:,0],psf,k)
img = Image.fromarray(b)
img.show()
b1 = b[:,362:413]
plt.imshow(b1, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = b1.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(b1[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.show()

# Restored images
c = my_gc_reg_toeplitz(b,psf,5,reg,tolerance)

d = c[:,362:413]
plt.imshow(d, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = d.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(d[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.show()

c = my_gc_reg_toeplitz(b,psf,10,reg,tolerance)

d = c[:,362:413]
plt.imshow(d, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = d.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(d[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.show()

c = my_gc_reg_toeplitz(b,psf,15,reg,tolerance)

d = c[:,362:413]
plt.imshow(d, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = d.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(d[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.savefig('Desktop/Magistrale/Python/restored_p_15', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

c = my_gc_reg_toeplitz(b,psf,20,reg,tolerance)

d = c[:,362:413]
plt.imshow(d, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
height, width = d.shape
for y in range(0, height, 3):
    for x in range(0, width, 3):
        plt.text(x, y, str(d[y, x]), color='black', fontsize=6, ha='center', va='center')

plt.show()


## REAL EXAMPLE
"""
# Read an image
A = np.array(Image.open('Desktop/Magistrale/Python/io.jpg'))
img = Image.fromarray(A)
img.show()
b = blur(A,psf,k)
c = gc_reg_toeplitz(b,psf,40,reg,tolerance)
image1 = Image.fromarray(b)
image2 = Image.fromarray(c)
image1.show()
image2.show()
"""