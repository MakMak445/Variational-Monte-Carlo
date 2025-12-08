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
        random_step = (np.random.rand() - 0.5) * 3
        x_prime = x + random_step
        p_x = (exact_wavefunc(x, n))**2
        p_x_prime = (exact_wavefunc(x_prime, n))**2
        if p_x != 0.:
            p_acc = min(p_x_prime / p_x, 1.)
        else: p_acc = 1.

        rand = np.random.rand()
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

def analytical_laplacian(theta, x, y, z):
    return (theta**2 - (2*theta / r(x,y,z))) * hydrogen_wavefunc(theta,x,y,z)

def local_energy_H_3D(theta, x, y, z):
    wf_value = hydrogen_wavefunc(theta, x, y, z)
    return (-0.5 * (cent_diff_4_laplacian(x,y,z,hydrogen_wavefunc,theta, h_der)/wf_value)) - 1/r(x,y,z)


def generate_randoms_3d(N_s, theta, burn_in = 100):
    pos_samples = np.zeros((N_s+burn_in, 3))
    pos_0 = np.array([1e-5,1e-5,1e-5])
    pos = pos_0
    for i in range(N_s + burn_in):
        random_x_step = (np.random.rand() - 0.5) * 2
        random_y_step = (np.random.rand() - 0.5) * 2
        random_z_step = (np.random.rand() - 0.5) * 2
        random_step = np.array([random_x_step, random_y_step, random_z_step])
        pos_prime = pos + random_step
        p_x = (hydrogen_wavefunc(theta, *pos))**2
        p_x_prime = (hydrogen_wavefunc(theta, *pos_prime))**2
        if p_x != 0.:
            p_acc = min(p_x_prime / p_x, 1.)
        else: p_acc = 1.

        rand = np.random.rand()
        if rand <= p_acc:
            pos = pos_prime
        pos_samples[i] = pos
    
    return pos_samples[burn_in:]

theta = 3.0 
samples = generate_randoms_3d(100000, theta)

r_samples = np.sqrt(np.sum(samples**2, axis=1))
#%%
'''
Numerical verification of Laplacian

'''
rand_x, rand_y, rand_z = np.random.rand(), np.random.rand(), np.random.rand()
theta_lap = np.random.rand() * 5
numerical_laplacian = cent_diff_4_laplacian(rand_x, rand_y, rand_z, hydrogen_wavefunc, theta_lap, 3e-3)
analyt_laplacian = analytical_laplacian(theta_lap, rand_x, rand_y, rand_z)
print(f"Absolute difference between analytical laplacian and numerically calculated laplacian is {abs(numerical_laplacian - analyt_laplacian)}, relative difference of {abs(numerical_laplacian - analyt_laplacian) / analyt_laplacian}")
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
    thetas = []
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
        thetas.append(theta)
        runs += 1
    final_samples = generate_randoms_3d(N_s, theta)
    final_x_vals = final_samples[:, 0]
    final_y_vals = final_samples[:, 1]
    final_z_vals = final_samples[:, 2]
    final_energies = local_energy_H_3D(theta, final_x_vals, final_y_vals, final_z_vals)
    final_E_exp = np.mean(final_energies)
    return theta, final_E_exp, runs, thetas, final_x_vals, final_y_vals, final_z_vals
#%%
'''
alphas = np.linspace(0.05, 2.1, 100)
runs_list = []
for alpha in alphas:
    try:
        theta, final_E_exp, runs_val, thetas, final_x_vals, final_y_vals, final_z_vals = optimise_theta(1000, 1e-9, 2., 1000, alpha)
        runs_list.append(runs_val)
        print(f'For alpha = {alpha}, theta converged to {theta} in {runs_val} runs with expected energy of {final_E_exp}')
    except Exception as e:
        print(f'Optimization failed for alpha = {alpha} with error: {e}')
'''
#%%
theta, final_E_exp, runs, thetas, final_x_vals, final_y_vals, final_z_vals = optimise_theta(1000, 1e-9, 2., 1000, 0.1)
print(theta, final_E_exp, runs)
plt.plot(range(len(thetas)), thetas, 'o-', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('Theta value')
plt.title('Theta Optimization Progression')
plt.grid()
plt.show()
#%%
'''
plt.plot(alphas, runs_list, 'o-', markersize=4)
plt.xlabel('Learning Rate (alpha)')
plt.ylabel('Number of Iterations to Converge')
plt.title('Effect of Learning Rate on Convergence Speed')
plt.grid()
plt.show()
'''
#%%
random_proof_samples = generate_randoms_3d(100000, theta)
final_x_vals = random_proof_samples[:, 0]
final_y_vals = random_proof_samples[:, 1]
final_z_vals = random_proof_samples[:, 2]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].hist2d(final_x_vals, final_y_vals, bins=(100,100),  density=True, cmap=plt.cm.inferno)
axs[0,0].set_title('100x100 Histogram in XY plane')

axs[0, 1].hist2d(final_x_vals, final_y_vals, bins=(50,50), density=True, cmap=plt.cm.inferno)
axs[0,1].set_title('50x50 Histogram in XY plane')

axs[1,0].hist2d(final_x_vals, final_y_vals, bins=(200,200), density=True, cmap=plt.cm.inferno)  
axs[1,0].set_title('200x200 Histogram in XY plane') 

axs[1,1].hist2d(final_x_vals, final_y_vals, bins=(400,400), density=True, cmap=plt.cm.inferno)
axs[1,1].set_title('400x400 Histogram in XY plane')
plt.tight_layout()
#plt.colorbar(ax=axs.ravel().tolist(), label='Density')

plt.show()
#%%

def r_vector(array):
    if array.ndim == 1:
        return np.sqrt(np.sum(array**2))
    return np.sqrt(np.sum(array**2, axis=1))

def h_molecule_wavefunc(theta, q1, q2, r1, r2):
    psi = (np.exp(-theta[0] * (r_vector(r1-q1) + r_vector(r2-q2))) + 
           np.exp(-theta[0] * (r_vector(r1-q2) + r_vector(r2-q1)))) * np.exp(-(theta[1]) / (1+theta[2]*r_vector(r1-r2)))
    return psi
def generate_randoms_3d_2particles(N_s, theta, q1, q2, r1_0 = np.array([1e-5,1e-5,1e-5]), r2_0 = -np.array([1e-5,1e-5,1e-5]), burn_in = 100):
    rng = np.random.default_rng()
    pos1_samples = np.zeros((N_s+burn_in, 3)) 
    pos2_samples = np.zeros((N_s+burn_in, 3))
    pos1_0 = r1_0
    pos2_0 = r2_0
    pos1 = pos1_0
    pos2 = pos2_0
    for i in range(N_s + burn_in):
        random1_step = rng.normal(0, 1, 3)
        pos1_prime = pos1 + random1_step

        random2_step = rng.normal(0, 1, 3)
        pos2_prime = pos2 + random2_step

        p_x = (h_molecule_wavefunc(theta, q1, q2, pos1, pos2))**2
        p_x_prime = (h_molecule_wavefunc(theta, q1, q2, pos1_prime, pos2_prime))**2
        if p_x != 0.:
            p_acc = min(p_x_prime / p_x, 1.)
        else: p_acc = 1.

        rand = np.random.rand()
        if rand <= p_acc:
            pos1 = pos1_prime
            pos2 = pos2_prime
        pos1_samples[i] = pos1
        pos2_samples[i] = pos2
    return pos1_samples[burn_in:], pos2_samples[burn_in:], pos1_samples[-1], pos2_samples[-1]

def generate_randoms_3d_vectorised(n_walkers, steps, step_size, theta, q1, q2, r1_0 = np.array([1e-5,1e-5,1e-5]), r2_0 = -np.array([1e-5,1e-5,1e-5]), burn_in = 100):
    rng = np.random.default_rng()
    if r1_0.ndim == 1:
        pos1 = np.tile(r1_0, (n_walkers, 1))
        pos2 = np.tile(r2_0, (n_walkers, 1))
    else:
        pos1 = r1_0.copy()
        pos2 = r2_0.copy()
    pos1_samples = np.zeros((steps, n_walkers, 3))
    pos2_samples = np.zeros((steps, n_walkers, 3))
    p_x = (h_molecule_wavefunc(theta, q1, q2, pos1, pos2))**2
    for i in range(steps + burn_in):
        random1_step = rng.normal(0, step_size, (n_walkers, 3))
        pos1_prime = pos1 + random1_step

        random2_step = rng.normal(0, step_size, (n_walkers, 3))
        pos2_prime = pos2 + random2_step
        p_x_prime = (h_molecule_wavefunc(theta, q1, q2, pos1_prime, pos2_prime))**2
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = p_x_prime / p_x
            np.nan_to_num(ratio, copy=False, nan=0.0, posinf=1.0)
            p_acc = np.minimum(ratio, 1.0)

        rand = rng.random(n_walkers)
        accepted = rand <= p_acc
        pos1[accepted] = pos1_prime[accepted]
        pos2[accepted] = pos2_prime[accepted]
        p_x[accepted] = p_x_prime[accepted]
        if i >= burn_in:
            pos1_samples[i - burn_in] = pos1
            pos2_samples[i - burn_in] = pos2
    return pos1_samples.reshape(-1, 3), pos2_samples.reshape(-1, 3), pos1_samples[-1], pos2_samples[-1]


def cent_diff_4_laplacian_H2(theta, q1, q2, r1, r2, func, h=1e-5):
    x_diff_h = np.array([h,0,0])
    y_diff_h = np.array([0,h,0])
    z_diff_h = np.array([0,0,h])

    x_diff_2h = np.array([2*h,0,0])
    y_diff_2h = np.array([0,2*h,0])
    z_diff_2h = np.array([0,0,2*h])

    current_point = func(theta, q1, q2, r1, r2)
    #loop for r1
    der_r1_x = (-(func(theta, q1, q2, r1 + x_diff_2h, r2) + func(theta, q1, q2, r1 - x_diff_2h, r2)) + 
            16 * (func(theta, q1, q2, r1 + x_diff_h, r2) + func(theta, q1, q2, r1 - x_diff_h, r2)) - 
            30 * current_point) / (12 * h**2)
    der_r1_y = (-(func(theta, q1, q2, r1 + y_diff_2h, r2) + func(theta, q1, q2, r1 - y_diff_2h, r2)) + 
            16 * (func(theta, q1, q2, r1 + y_diff_h, r2) + func(theta, q1, q2, r1 - y_diff_h, r2)) - 
            30 * current_point) / (12 * h**2)
    der_r1_z = (-(func(theta, q1, q2, r1 + z_diff_2h, r2) + func(theta, q1, q2, r1 - z_diff_2h, r2)) + 
            16 * (func(theta, q1, q2, r1 + z_diff_h, r2) + func(theta, q1, q2, r1 - z_diff_h, r2)) - 
            30 * current_point) / (12 * h**2)
    #loop for r2
    der_r2_x = (-(func(theta, q1, q2, r1, r2 + x_diff_2h) + func(theta, q1, q2, r1, r2 - x_diff_2h)) + 
            16 * (func(theta, q1, q2, r1, r2 + x_diff_h) + func(theta, q1, q2, r1, r2 - x_diff_h)) - 
            30 * current_point) / (12 * h**2)
    der_r2_y = (-(func(theta, q1, q2, r1, r2 + y_diff_2h) + func(theta, q1, q2, r1, r2 - y_diff_2h)) + 
            16 * (func(theta, q1, q2, r1, r2 + y_diff_h) + func(theta, q1, q2, r1, r2 - y_diff_h)) - 
            30 * current_point) / (12 * h**2)
    der_r2_z = (-(func(theta, q1, q2, r1, r2 + z_diff_2h) + func(theta, q1, q2, r1, r2 - z_diff_2h)) + 
            16 * (func(theta, q1, q2, r1, r2 + z_diff_h) + func(theta, q1, q2, r1, r2 - z_diff_h)) -
            30 * current_point) / (12 * h**2)
    return der_r1_x + der_r1_y + der_r1_z + der_r2_x + der_r2_y + der_r2_z

def cent_diff_2_laplacian_H2(theta, q1, q2, r1, r2, func, h=1e-5):
    x_diff_h = np.array([h,0,0])
    y_diff_h = np.array([0,h,0])
    z_diff_h = np.array([0,0,h])

    #x_diff_2h = np.array([2*h,0,0])
    #y_diff_2h = np.array([0,2*h,0])
    #z_diff_2h = np.array([0,0,2*h])

    current_point = func(theta, q1, q2, r1, r2)
    #loop for r1
    der_r1_x =   (func(theta, q1, q2, r1 + x_diff_h, r2) + func(theta, q1, q2, r1 - x_diff_h, r2) - 
                  2 * current_point) / (h**2)
    der_r1_y = (func(theta, q1, q2, r1 + y_diff_h, r2) + func(theta, q1, q2, r1 - y_diff_h, r2) - 
                  2 * current_point) / (h**2)
    der_r1_z = (func(theta, q1, q2, r1 + z_diff_h, r2) + func(theta, q1, q2, r1 - z_diff_h, r2) - 
                  2 * current_point) / (h**2)
    #loop for r2
    der_r2_x = (func(theta, q1, q2, r1, r2 + x_diff_h) + func(theta, q1, q2, r1, r2 - x_diff_h) - 
                  2 * current_point) / (h**2)
    der_r2_y = (func(theta, q1, q2, r1, r2 + y_diff_h) + func(theta, q1, q2, r1, r2 - y_diff_h) - 
                  2 * current_point) / (h**2)
    der_r2_z = (func(theta, q1, q2, r1, r2 + z_diff_h) + func(theta, q1, q2, r1, r2 - z_diff_h) -
            2 * current_point) / (h**2)
    return der_r1_x + der_r1_y + der_r1_z + der_r2_x + der_r2_y + der_r2_z

def local_energy_H2(theta, q1, q2, r1, r2):
    wf_value = h_molecule_wavefunc(theta, q1, q2, r1, r2)
    kinetic = -0.5 * cent_diff_2_laplacian_H2(theta, q1, q2, r1, r2, h_molecule_wavefunc) / wf_value
    potential = -(1/r_vector(r1-q1) + 1/r_vector(r1-q2) + 
                  1/r_vector(r2-q1) + 1/r_vector(r2-q2)) + (1/r_vector(r1 - r2) + 1/np.linalg.norm(q1 - q2))
    return kinetic + potential
#%%

def der_H2_theta(theta, q1, q2, r1, r2, energies, E_exp):
    N_s = energies.shape[0]
    dphitheta = np.zeros((3, N_s))
    wf_value = h_molecule_wavefunc(theta, q1, q2, r1, r2)
    r1q1 = abs(r_vector(r1 - q1))
    r1q2 = abs(r_vector(r1 - q2))
    r2q1 = abs(r_vector(r2 - q1))
    r2q2 = abs(r_vector(r2 - q2))
    r2r1 = abs(r_vector(r2 - r1))
    non_cross = r1q1+r2q2
    cross = r1q2 + r2q1
    #Analytically found derivative wrt theta[0]
    part_der_0 = (-non_cross * np.exp(-theta[0] * non_cross) - cross * np.exp(-theta[0] * cross)) * np.exp(-(theta[1])/(1+theta[2]*r2r1))
    #Analytically found derivative wrt theta[1]
    part_der_1 = (-1/(1 + theta[2] * r2r1)) * wf_value
    #Analytically found derivative wrt theta[2]
    part_der_2 = ((r2r1 * theta[1]) / (1 + theta[2]*r2r1)**2) * wf_value

    dphitheta[0] = part_der_0
    dphitheta[1] = part_der_1
    dphitheta[2] = part_der_2

    grad_exp = ((energies - E_exp) * (dphitheta/wf_value))
    return 2 * np.mean(grad_exp, axis=1)

def iterate_theta_H2(theta, q1, q2, r1, r2, alpha, exp_E, energies):
    new_theta = theta - alpha * der_H2_theta(theta, q1, q2, r1, r2, energies, exp_E)
    return new_theta

def optimise_theta_H2(N_s, convergence_ratio, theta_0, max_runs, alpha, q1, q2, minruns = 100):
    theta = theta_0
    thetas = []
    runs = 0
    theta_change_ratio = 100
    while (theta_change_ratio > convergence_ratio and  runs <= max_runs) or runs < minruns:
        if runs == 0:
            r1_samples, r2_samples, last_r1, last_r2 = generate_randoms_3d_vectorised(N_s, 1000, 0.5, theta, q1, q2)
        else:
            r1_samples, r2_samples, last_r1, last_r2 = generate_randoms_3d_vectorised(N_s, 1000, 0.5, theta, q1, q2, last_r1, last_r2)
        energies = local_energy_H2(theta, q1, q2, r1_samples, r2_samples)
        exp_E_H = np.mean(energies)
        theta_prime = iterate_theta_H2(theta, q1, q2, r1_samples, r2_samples, alpha, exp_E_H, energies)
        theta_change_ratio = np.max(abs((theta - theta_prime) / theta))
        theta = theta_prime
        thetas.append(theta)
        runs += 1
        print(runs)
    final_r1_samples, final_r2_samples, last_r1, last_r2 = generate_randoms_3d_vectorised(N_s, 1000, 0.5, theta, q1, q2, last_r1, last_r2)
    final_energies = local_energy_H2(theta, q1, q2, final_r1_samples, final_r2_samples)
    final_E_exp = np.mean(final_energies)

    return theta, final_E_exp, runs, thetas, final_r1_samples, final_r2_samples
#%%
Ns = 10
alpha = 0.01
theta_0 = [1.0, 1., 1.]  # Initial guess for theta

qs = np.linspace(1e-2, 2., 10)  # Nuclear separations from near 0 to 2.5 a.u.
energies = []
nuclear_separations = []
test_samples_r1, test_samples_r2, _, _ = generate_randoms_3d_vectorised(Ns, 1000, 0.5, theta_0, np.array([-1.,0.,0.]), np.array([1.,0.,0.]))
all_test_samples = np.vstack((test_samples_r1, test_samples_r2))
test_x_vals = all_test_samples[:, 0]
test_y_vals = all_test_samples[:, 1]
plt.hist2d(test_x_vals, test_y_vals, bins=500, density=True, cmap=plt.cm.inferno, range=[[-4, 4], [-4, 4]])
plt.title(f'Histogram in XY plane for nuclear separation of 2.0')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
#'''
#%%
for index, q in enumerate(qs):
    q1, q2 = np.array([q, 0., 0.]), np.array([-q, 0., 0.])
    theta, E_exp, runs, thetas, final_r1_samples, final_r2_samples = optimise_theta_H2(Ns, 5e-5, theta_0, 1000, alpha, q1, q2)
    energies.append(E_exp)
    nuclear_separations.append(2*q)
    print(f'For nuclear separation of {abs(q1 - q2)}, theta converged to {theta} in {runs} runs with expected energy of {E_exp} Hartree')
    all_samples = np.vstack((final_r1_samples, final_r2_samples))
    x_vals = all_samples[:, 0]
    y_vals = all_samples[:, 1]
    plt.hist2d(x_vals, y_vals, bins=200, density=True, cmap=plt.cm.inferno, range=[[-4, 4], [-4, 4]])
    plt.title(f'Histogram in XY plane for nuclear separation of {2*q}')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.plot(range(len(thetas)), [t[0] for t in thetas], 'o-', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Theta[0] value')
    plt.title(f'Theta[0] Optimization Progression for nuclear separation of {abs(q1 - q2)}')
    plt.grid()
    plt.show()
#%%
plt.plot(nuclear_separations, energies, 'o-', markersize=4)
plt.xlabel('Nuclear Separation (a.u.)')
plt.ylabel('Expected Energy (Hartree)')
plt.title('H2 Molecule Energy vs Nuclear Separation')
plt.axhline(y=-1.174, color='r', linestyle='--', label='Target (-1.174 Hartree)')
plt.legend()
plt.show()

'''


# --- VERIFICATION PARAMETERS ---

q1_test = np.array([0., 0., 0.7])  # Nucleus A
q2_test = np.array([0., 0., -0.7]) # Nucleus B (R = 1.4)

# Safe starting guess
theta_start = np.array([1.0, 0.5, 0.1])

# Conservative learning rate
learning_rate = 0.001 

print("--- STARTING H2 EQUILIBRIUM TEST ---")
print(f"Target Energy: approx -1.17 Hartree")

theta_final, E_final, runs, hist_theta, r1_fin, r2_fin = optimise_theta_H2(
    N_s=1000, 
    convergence_ratio=1e-5, 
    theta_0=theta_start, 
    max_runs=1000, 
    alpha=learning_rate, 
    q1=q1_test, 
    q2=q2_test
)
print(f"Converged in {runs} iterations")
print(f"Final Theta: {theta_final}")
print(f"Final Energy: {E_final}")

# --- DIAGNOSTIC PLOT ---
plt.plot(range(len(hist_theta)), [t[0] for t in hist_theta], 'o-', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('Theta[0] value')
plt.title('Theta[0] Optimization Progression for H2 Molecule')
plt.grid()
plt.show()
# 1. Plot Energy Convergence
# We want to see a "Hockey Stick" curve that flattens out
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
# Note: You'll need to return the energy history from your optimizer 
# or just plot the final printouts. Assuming you add 'energy_history' to returns:
# plt.plot(energy_history) 
plt.title("Did Energy Stabilize?")
plt.xlabel("Iterations")
plt.ylabel("Energy (Hartree)")
plt.axhline(y=-1.174, color='r', linestyle='--', label='Target (-1.174)')
plt.legend()

# 2. Plot Electron Density
# This confirms electrons are clustering around the nuclei
plt.subplot(1,2,2)
plt.hist2d(r1_fin[:,0], r1_fin[:,2], bins=50, cmap='plasma')
plt.plot([0,0], [0.7, -0.7], 'wo', markersize=5, label='Nuclei') # Mark Nuclei
plt.title("Electron Density (XZ Plane)")
plt.xlabel("X")
plt.ylabel("Z")
plt.legend()
plt.tight_layout()
plt.show()
'''
# %%
