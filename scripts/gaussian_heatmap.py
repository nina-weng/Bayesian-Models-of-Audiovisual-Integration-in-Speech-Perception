import numpy as np
import matplotlib.pyplot as plt
from fitting_functions import *
from scipy.stats import multivariate_normal
from scipy.stats import norm

# fitted_param_path = '../S2_data/fitted_params_6.npy'
# fitted_param_path = '../fitted_params/fitted_params_11.npy'
fitted_param_path = '../fitted_params/fitted_params_bci_full_1.npy'
params_stored= np.load(fitted_param_path)
print(params_stored.shape)


def plot_distribution( density_fun, color=None, visibility=1, label=None):
    plt_range = 5

    # create grid for parameters (a,b)
    num_points = 300
    a_array = np.linspace(-plt_range, plt_range, num_points)
    b_array = np.linspace(-plt_range, plt_range, num_points)
    A_array, B_array = np.meshgrid(a_array, b_array)

    # form array with all combinations of (a,b) in our grid
    AB = np.column_stack((A_array.ravel(), B_array.ravel()))

    # evaluate density for every point in the grid and reshape bac
    Z = density_fun(A_array.ravel(), B_array.ravel())

    Z = Z.reshape((len(a_array), len(b_array)))
    #     print(Z[:,0])

    if color is None:
        #         plt.contour(a_array, b_array, np.exp(Z), alpha=visibility)
        plt.pcolormesh(a_array, b_array, np.exp(Z), shading='auto', cmap=plt.cm.viridis)  # YlGnBu, viridis
    else:
        #         plt.contour(a_array, b_array, np.exp(Z), colors=color, alpha=visibility)
        plt.pcolormesh(a_array, b_array, np.exp(Z), shading='auto', cmap=plt.cm.viridis, alpha=visibility)
    #         plt.plot(a_array,np.sum(np.exp(Z),axis=0)/np.sum(np.exp(Z))*50)
    plt.xlabel('auditory')
    plt.ylabel('visual');

    if label:
        plt.plot([-1000], [-1000], color=color, label=label)
    plt.xlim((-(plt_range - 1), (plt_range - 1)))
    plt.ylim((-(plt_range - 1), (plt_range - 1)))




test_index = 4
x0 = params_stored[test_index,:]
# sigma_0_s, sigma_0_a = x0[0:2]
[mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
[sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
c = x0[12:]

print(x0)

# sigma_0 = sigma_0_s

# compute the mu and sigma for av
mu_avs_pc1,sigma_avs_pc1 = get_params_AV(mu_ab,mu_vg,sigma_vh,sigma_vm,sigma_vl,sigma_ah,sigma_am,sigma_al,sigma_0=0)
mu_avs_pc2,sigma_avs_pc2 = get_params_AV(mu_ab,mu_vg,sigma_vh,sigma_vm,sigma_vl,sigma_ah,sigma_am,sigma_al,sigma_0=np.inf)
# print(mu_avs,sigma_avs)

# generate stimulate data from mu and sigma
sample_size = 200
mu_abs = [mu_ab]*5
sigma_abs = np.array([sigma_ah,sigma_ah,sigma_ah,sigma_am,sigma_al])
audio_stimulateds = np.array([np.random.normal(mu_abs[i], sigma_abs[i], sample_size) for i in range(5)])
print(audio_stimulateds.shape)


mu_vgs = [mu_vg]*5
sigma_vgs = np.array([sigma_vh,sigma_vm,sigma_vl,sigma_vh,sigma_vh])
visual_stimulateds = np.array([np.random.normal(mu_vgs[i], sigma_vgs[i], sample_size) for i in range(5)])

audiovisual_stimulateds = np.array([np.random.normal(mu_avs[i], sigma_avs[i], sample_size) for i in range(5)])

# plot

fig, axs = plt.subplots(5, 1, figsize=(12, 12))

for i in range(5):
    ax_c = axs[i]

    audio_stimulated = list(audio_stimulateds[i])
    visual_stimulated = list(visual_stimulateds[i])
    av_stimulated = list(audiovisual_stimulateds[i])

    min_ = min(min(audio_stimulated), min(visual_stimulated), min(av_stimulated))
    max_ = max(max(audio_stimulated), max(visual_stimulated), max(av_stimulated))
    bins = np.linspace(min_, max_, 70)
    x = np.linspace(min_, max_, 1000)

    ax_c.hist(audio_stimulated, bins=bins, color='lightcoral', alpha=0.6, edgecolor='lightcoral', density=True,
              label='audio stimulated data')
    ax_c.plot(x, norm(mu_abs[i], sigma_abs[i]).pdf(x), color='lightcoral', linewidth=2.0, label='audio fitted guassian')

    ax_c.hist(visual_stimulated, bins=bins, color='steelblue', alpha=0.6, edgecolor='steelblue', density=True,
              label='visual stimulated data')
    ax_c.plot(x, norm(mu_vgs[i], sigma_vgs[i]).pdf(x), color='steelblue', linewidth=2.0, label='visual fitted guassian')

    ax_c.hist(av_stimulated, bins=bins, color='olive', alpha=0.6, edgecolor='steelblue', density=True,
              label='audiovisual stimulated data')
    ax_c.plot(x, norm(mu_avs[i], sigma_avs[i]).pdf(x), color='olive', linewidth=2.0,
              label='audiovisual fitted guassian')

    ax_c.vlines(c[0], ymin=0, ymax=0.5, color='black')
    ax_c.vlines(c[1], ymin=0, ymax=0.5, color='red')
    ax_c.set_xlim((-5, 5))

plt.show()

# hyperparameters of the prior distribution
alpha = 1
m0 = np.array([0, 0])
rita = 10
sigma = np.abs(sigma_0)
S0 = np.array([[rita, rita - sigma],
               [rita - sigma, rita]])

m1 = np.array([mu_ab, mu_vg])
S1 = np.array([[sigma_am, 0.],
               [0., sigma_vl]])



# mu_av,sigma_av = get_av(mu_ab,mu_vg,sigma_ah,sigma_vh,sigma_0=sigma_0_s)



# hyperparameters of the likelihood
beta = 1 / 2

# log normal density
log_npdf = lambda x, m, v: -0.5 * np.log(2 * np.pi * v) - 0.5 * (x - m) ** 2 / v

predict = lambda x, a, b: a + b * x


def log_prior( a, b):
    return multivariate_normal.logpdf(np.column_stack([a, b]), m0, S0)


def log_likelihood( a, b):
    return multivariate_normal.logpdf(np.column_stack([a, b]), m1, S1)


# def log_posterior( a, b):
#     return multivariate_normal.logpdf(np.column_stack([a, b]), m2, S2)

def log_posterior(a, b):
    return log_prior( a, b) + log_likelihood( a, b)


# plot
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plot_distribution(density_fun=log_prior, color='b')
plt.title('Prior')

plt.subplot(1, 3, 2)
plot_distribution(density_fun=log_likelihood, color='r')
plt.title('Likelihood')

plt.subplot(1, 3, 3)
# plot_distribution(xtrain, ttrain, density_fun=log_prior, color='b', visibility=0.25, label='Prior')
# plot_distribution(xtrain, ttrain, density_fun=log_likelihood, color='r', visibility=0.25, label='Likelihood')
plot_distribution(density_fun=log_posterior, color='g', label='Posterior')
plt.title('Posterior')
plt.show()