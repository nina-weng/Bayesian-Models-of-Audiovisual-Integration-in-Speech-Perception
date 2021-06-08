import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from fitting_functions import *
import pickle

PLOT_RANGE = 8
ALPHA = 0.3
FONTSIZE= 10
TEXT_Y = [0.35,0.4,0.48]

test_index = 5
chosen_type = 4
print('tester index:{}'.format(test_index))

fitted_param_path_bci = '../fitted_params/fitted_params_bci_full_1.npy'
fitted_param_path_jpm = '../fitted_params/fitted_params_jpm_full_1.npy'
params_stored_bci = np.load(fitted_param_path_bci)
params_stored_jpm = np.load(fitted_param_path_jpm)
# print(fitted_param_path_bci.shape)
print('bci parameters from :{}'.format(fitted_param_path_bci))
print('jpm parameters from :{}'.format(fitted_param_path_jpm))

pkl_path = '../S2_data/data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)


x0 = params_stored_bci[test_index,:]
p_s,p_a = x0[0:2]
[mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
[sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
c = x0[12:]

x0_jpm = params_stored_jpm[test_index,:]
sigma_0_s,sigma_0_a = x0_jpm[0:2]
[mu_vg_jpm, mu_vb_jpm, mu_ag_jpm, mu_ab_jpm] = x0_jpm[2:6]
[sigma_vh_jpm, sigma_vm_jpm, sigma_vl_jpm, sigma_ah_jpm, sigma_am_jpm, sigma_al_jpm] = x0_jpm[6:12]
c_jpm = x0_jpm[12:]


# BCI
mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)



res_prob_pc1 = guassian2prob(mus_pc1, sigmas_pc1, c)
res_prob_pc2 = guassian2prob(mus_pc2, sigmas_pc2, c)

res_prob_s = p_s*res_prob_pc1 + (1-p_s)*res_prob_pc2
res_prob_a = p_a*res_prob_pc1 + (1-p_a)*res_prob_pc2


# JPM
mus_jpm_s, sigmas_jpm_s = get_params_AV(mu_ab_jpm, mu_vg_jpm, sigma_vh_jpm, sigma_vm_jpm,
                                sigma_vl_jpm, sigma_ah_jpm, sigma_am_jpm,
                                sigma_al_jpm,
                                sigma_0=sigma_0_s)
mus_jpm_a, sigmas_jpm_a = get_params_AV(mu_ab_jpm, mu_vg_jpm, sigma_vh_jpm, sigma_vm_jpm,
                                sigma_vl_jpm, sigma_ah_jpm, sigma_am_jpm,
                                sigma_al_jpm,
                                sigma_0=sigma_0_a)
res_prob_jpm_s = guassian2prob(mus_jpm_s, sigmas_jpm_s, c_jpm)
res_prob_jpm_a = guassian2prob(mus_jpm_a, sigmas_jpm_a, c_jpm)


########################################


#
# fig, axs = plt.subplots(5, 2, figsize=(12, 12))

plt.figure(figsize=(12,6))

j=1

if j == 0:
    mus_jpm = mus_jpm_s
    sigmas_jpm = sigmas_jpm_s
    res_prob_jpm = res_prob_jpm_s
    ori_prob = np.array(data['AVFus']['synch']['props'][test_index, :, :])
    p = p_s
    # title_str = sub_plot_title[i]+' synchrony'
    res_prob = res_prob_s
else:
    mus_jpm = mus_jpm_a
    sigmas_jpm = sigmas_jpm_a
    res_prob_jpm = res_prob_jpm_a
    ori_prob = np.array(data['AVFus']['asynch']['props'][test_index, :, :])
    p = p_a
    # title_str = sub_plot_title[i]+' asynchrony'
    res_prob = res_prob_a

xs = np.linspace(-PLOT_RANGE,PLOT_RANGE,100)
plt.plot(xs, stats.norm.pdf(xs, mus_jpm[chosen_type], sigmas_jpm[chosen_type]),
               color='mediumorchid', linewidth=3, label='JPM')
plt.plot(xs, p*stats.norm.pdf(xs, mus_pc1[chosen_type], sigmas_pc1[chosen_type])+
                      (1-p)*stats.norm.pdf(xs, mus_pc2[chosen_type], sigmas_pc2[chosen_type]),
                    color='limegreen',linewidth=3,label='BCI')


# generating samples




# plt.xlim(-5,8)
plt.legend()
plt.show()