import numpy as np
import matplotlib.pyplot as plt
import cmath

def create_set(min,max, n):
    x=np.linspace(min,max,n)
    y=np.sin(x)
    return x,y
def plot_set(x,y):
    plt.plot(x,y, 'x')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Plot of sin(x)')
    plt.grid(True)
    plt.show()


def lagrange_poly(x, x_data, y_data):
    n = len(y_data)
    y = 0
    for i in range(n):
        coeff_i = 1
        for j in range(n):
            if i != j:
                coeff_i *= (x-x_data[j])/(x_data[i]-x_data[j])
        y += coeff_i * y_data[i]
    return y

'''
x_in, y_in = create_set(0, 1, 4)
xs = np.linspace(0, 1, 50)
ys = [lagrange_poly(x,x_in, y_in) for x in xs]
plt.plot(x_in, y_in, 'x', markersize=7, color='red')
plt.plot(xs, ys, color='blue')
plt.show()
'''


def disc_fourier(ys):
    N = len(ys)
    fouriered = []
    for i in range(N):
        f_p = 0
        for p in range(N):
            f_p+=ys[p]*cmath.exp(1j*2*np.pi*i*p/N)
        fouriered.append(f_p)
    return fouriered

def create_freq_domain(xs):
    """
    Calculates the frequency bins for a DFT based on the
    original signal's domain (xs).
    """
    N = len(xs)
    d = xs[1] - xs[0]  # Sample spacing
    L = N * d          # Total length of the signal
    
    freqs = []
    for k in range(N):
        if k < N / 2:
            # This is a positive frequency
            f_k = k / L
        else:
            # This is a negative frequency (wrapped around)
            f_k = (k - N) / L
        freqs.append(f_k)
        
    return freqs

xs = np.linspace(0, 2*np.pi, 1000)
ys = np.sin(xs)
fouriered = disc_fourier(ys)
plt.plot(xs, ys)
plt.show()
plt.plot(create_freq_domain(xs), fouriered)
plt.show()


