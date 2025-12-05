#%%
import numpy as np
from scipy.special import eval_hermite
import matplotlib.pyplot as plt
import random
import math
#%%
'''
herms = []
for i in range(8):
    herms.append(hermite(i))

x = np.linspace(-2, 2, 10000)
ys = []
for herm in herms:
    ys.append(herm(x))
n = 0
for y in ys:
    plt.plot(x,y, label = n)
    n+=1
plt.legend()
plt.show()
'''
'''
We assume that wavefunction spans across all space (domain stretches from -inf to inf), thus central difference scheme is best to numerically determine derivative

We use a central difference scheme that is accurate to 2nd order
'''
#%%
def exact_wavefunc(x, n):
    return eval_hermite(n, x) * np.exp(-x**2 / 2)

def exact_der(x, n):
    return (x**2 - 2*n - 1) * exact_wavefunc(x, n)

def cent_diff_2(x, n, h, func):
    '''
    h is the step size
    
    '''
    return (func(x+h, n) - 2 * func(x, n) + func(x-h, n)) / h**2

def cent_diff_4(x, n, h, func):
    return (-(func(x+2*h, n) + func(x-2*h, n)) + 
            16 * (func(x+h, n) + func(x-h, n)) - 
            30 * func(x, n)) / (12 * h**2)

def cent_diff_6(x,n,h, func):
    return (2 * (func(x+3*h, n) + func(x-3*h, n)) - 
            27 * (func(x+2*h, n) + func(x-2*h, n)) + 
            270 * (func(x+h, n) + func(x-h, n)) - 
            490 * func(x, n)) / (180 * h**2)

def cent_diff_8(x,n,h, func):
    return (-9 * (func(x+4*h, n) + func(x-4*h, n)) +
            128 * (func(x+3*h, n) + func(x-3*h, n)) - 
            1008 * (func(x+2*h, n) + func(x-2*h, n)) + 
            8064 * (func(x+h, n) + func(x-h, n)) - 
            14350 * func(x, n)) / (5040 * h**2)
#%%
schemes = [cent_diff_2, cent_diff_4, cent_diff_6, cent_diff_8]
x = 5
n = 7
h = np.linspace(1e-7, 1, 1000)
numerical_ders = []
for scheme in schemes:
    numerical_ders.append(scheme(x, n, h, exact_wavefunc)) 

j = 1
exact = exact_der(x, n)
for der in numerical_ders:
    plt.plot(h, abs((der - exact) / exact), label = f'order {2*j}')
    j+=1

plt.ylabel('relative error in derivative')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('step size')
plt.vlines(1.75e-2, 0, 1, colors= 'red')
plt.legend()
plt.grid()
plt.show()
#%%
'''
After seeing graph, choosing 4th order difference scheme at h = 3e-3
'''



h_der = 3e-3
def generate_randoms(N_s, n):
    x_samples = np.zeros(N_s)
    x_0 = 0.
    x = x_0
    for i in range(N_s):
        random_step = (random.random() - 0.5) * 3
        x_prime = x + random_step
        p_x = (exact_wavefunc(x, n))**2
        p_x_prime = (exact_wavefunc(x_prime, n))**2
        if p_x != 0.:
            p_acc = min(p_x_prime / p_x, 1.)
        else: p_acc = 1.

        rand = random.random()
        if rand <= p_acc:
            x = x_prime
        x_samples[i] = x
    norm_const = np.sqrt(np.pi) * 2**n * math.factorial(n) #analytically derived
    x_uniform = np.linspace(np.min(x_samples), np.max(x_samples), 10000)
    probs = exact_wavefunc(x_uniform, n)**2 / norm_const
    return x_samples, x_uniform, probs
#%%
n = 0
N_s = 100000
#%%
x, x_uniform, probs = generate_randoms(100000, n)
plt.figure(figsize=(10, 6))

plt.hist(x, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.plot(x_uniform, probs, '--',color = 'red')
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.xlabel('Position x')
plt.ylabel('Probability Density')
plt.title('Histogram of Metropolis Samples')

plt.show()
#%%
local_energies = -0.5 * (1 / exact_wavefunc(x, n)) * cent_diff_4(x, n, h_der, exact_wavefunc) + 0.5 * x**2
exp_E = np.mean(local_energies)
std_E = np.std(local_energies)
E_to_exp_diff = (exp_E - (n + 0.5))

print(f'expected E is {exp_E}, with standard deviation of {std_E}, difference between actual and expected is {E_to_exp_diff}')
#%%
def r(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)
def hydrogen_wavefunc(theta, x, y, z):
    return np.exp(-theta * r(x,y,z))

def cent_diff_4_laplacian(x, y, z, func, theta, h):
    current_point = func(theta, x, y, z)
    der_x = (-(func(theta, x+2*h, y, z) + func(theta, x-2*h, y, z)) + 
            16 * (func(theta, x+h, y, z) + func(theta, x-h, y, z)) - 
            30 * current_point) / (12 * h**2)
    der_y = (-(func(theta, x, y+2*h, z) + func(theta, x, y-2*h, z)) + 
            16 * (func(theta, x, y+h, z) + func(theta, x, y-h, z)) - 
            30 * current_point) / (12 * h**2)
    der_z = (-(func(theta, x, y, z+2*h) + func(theta, x, y, z-2*h)) + 
            16 * (func(theta, x, y, z+h) + func(theta, x, y, z-h)) - 
            30 * current_point) / (12 * h**2)

    return der_x + der_y + der_z

def local_energy_H_3D(theta, x, y, z):
    wf_value = hydrogen_wavefunc(theta, x, y, z)
    return (-0.5 * (cent_diff_4_laplacian(x,y,z,hydrogen_wavefunc,theta, h_der)/wf_value)) - 1/r(x,y,z)


def generate_randoms_3d(N_s, theta, burn_in = 1000):
    pos_samples = np.zeros((N_s+burn_in, 3))
    pos_0 = np.array([1e-5,1e-5,1e-5])
    pos = pos_0
    for i in range(N_s + burn_in):
        random_x_step = (random.random() - 0.5) * 2
        random_y_step = (random.random() - 0.5) * 2
        random_z_step = (random.random() - 0.5) * 2
        random_step = np.array([random_x_step, random_y_step, random_z_step])
        pos_prime = pos + random_step
        p_x = (hydrogen_wavefunc(theta, *pos))**2
        p_x_prime = (hydrogen_wavefunc(theta, *pos_prime))**2
        if p_x != 0.:
            p_acc = min(p_x_prime / p_x, 1.)
        else: p_acc = 1.

        rand = random.random()
        if rand <= p_acc:
            pos = pos_prime
        pos_samples[i] = pos
    
    return pos_samples[burn_in:]

theta = 3.0 
samples = generate_randoms_3d(100000, theta)

r_samples = np.sqrt(np.sum(samples**2, axis=1))
#%%
plt.figure(figsize=(10, 6))
plt.hist(r_samples, bins=100, density=True, alpha=0.6, label='Metropolis Samples', color='skyblue', edgecolor='black')

r_plot = np.linspace(0, np.max(r_samples), 200)
analytical_curve = 4 * (theta**3) * (r_plot**2) * np.exp(-2 * theta * r_plot)

plt.plot(r_plot, analytical_curve, 'r-', linewidth=2.5, label=r'Analytical $P(r) \propto r^2 |\psi|^2$')

plt.xlabel('Radius $r$')
plt.ylabel('Radial Probability Density')
plt.title(f'Radial Distribution Verification ($\\theta={theta}$)')
plt.legend()
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.show()


#def E_exp_der(N_s)
#%%

def der_H_theta(exp_E, energies, samples):
    r_samples = np.sqrt(np.sum(samples**2, axis = 1))
    differences = energies - exp_E
    return 2 * np.mean(-r_samples * differences)

def iterate_theta(theta, alpha, exp_E, energies, samples):
    new_theta = theta - alpha * der_H_theta(exp_E, energies, samples)
    return new_theta

def optimise_theta(N_s, convergence_ratio, theta_0, max_runs, alpha):
    theta = theta_0
    runs = 0
    theta_change_ratio = 100
    while theta_change_ratio > convergence_ratio and runs<= max_runs:
        samples = generate_randoms_3d(N_s, theta)
        x_vals = samples[:, 0]
        y_vals = samples[:, 1]
        z_vals = samples[:, 2]
        energies = local_energy_H_3D(theta, x_vals, y_vals, z_vals)
        exp_E_H = np.mean(energies)
        theta_prime = iterate_theta(theta, alpha, exp_E_H, energies, samples)
        theta_change_ratio = abs((theta - theta_prime) / theta)
        theta = theta_prime
        runs += 1
    return theta, exp_E_H

print(optimise_theta(1000, 1e-9, 2., 100, 0.05))