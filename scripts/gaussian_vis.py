
import matplotlib.pyplot as plt
from fitting_functions import *
import scipy.stats as stats
import pickle


PLOT_RANGE = 6
ALPHA = 0.3
FONTSIZE= 10
TEXT_Y = [0.35,0.4,0.48]




# 5,6,7,12 - more obvious 2-peak (dissimilar)
# 1,3,10,12 - DA prob ~ 0
# 0,2,4,8,9,11,13,14,15 - quite similar with the other model
test_index = 12
print('tester index:{}'.format(test_index))

fitted_param_path_bci = '../fitted_params/fitted_params_bci_full_4.npy'
fitted_param_path_jpm = '../fitted_params/fitted_params_jpm_full_4.npy'
params_stored_bci = np.load(fitted_param_path_bci)
params_stored_jpm = np.load(fitted_param_path_jpm)
# print(fitted_param_path_bci.shape)
print('bci parameters from :{}'.format(fitted_param_path_bci))
print('jpm parameters from :{}'.format(fitted_param_path_jpm))


pkl_path = '../S2_data/data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)


# print(data['AVFus']['synch']['snr'])

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

# AVFus - audio Ba, visual Ga

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



# PLOT

fig, axs = plt.subplots(5, 2, figsize=(12, 12))
sub_plot_title = ['Vhi_Ahi','Vmid_Ahi','Vlo_Ahi','Vhi_Amid','Vhi_Alo']

for i in range(5):
    xs = np.linspace(-PLOT_RANGE,PLOT_RANGE,100)

    for j in range(2):

        ############
        # BCI Part #
        ############

        if j == 0:
            p = p_s
            # title_str = sub_plot_title[i]+' synchrony'
            res_prob= res_prob_s
        else:
            p = p_a
            # title_str = sub_plot_title[i]+' asynchrony'
            res_prob = res_prob_a


        axs[i,j].plot(xs,stats.norm.pdf(xs, mus_pc1[i], sigmas_pc1[i]),
                    color='gold',linestyle ='--',linewidth = 2,label='P(C=1)')
        axs[i,j].plot(xs,stats.norm.pdf(xs, mus_pc2[i], sigmas_pc2[i]),
                    color='seagreen',linestyle ='--',linewidth = 2,label='P(C=2)')
        axs[i,j].plot(xs, p*stats.norm.pdf(xs, mus_pc1[i], sigmas_pc1[i])+
                      (1-p)*stats.norm.pdf(xs, mus_pc2[i], sigmas_pc2[i]),
                    color='limegreen',linewidth=3,label='BCI')

        xs_part = np.linspace(c[0],c[1],100)
        ys_part = p*stats.norm.pdf(xs_part, mus_pc1[i], sigmas_pc1[i])+\
                  (1-p)*stats.norm.pdf(xs_part, mus_pc2[i], sigmas_pc2[i])
        axs[i,j].fill_between(xs_part, ys_part, color='limegreen',alpha=ALPHA,
                              label='BCI - DA area',hatch='...')

        # print the prob on the plot
        axs[i,j].text(c[0]/2 - PLOT_RANGE/2, TEXT_Y[0], s='{:.2f}'.format(res_prob[i,0]),
                      fontsize=FONTSIZE,ha='center',va='center',color='limegreen')
        axs[i,j].text(c[0] / 2 + c[1] / 2, TEXT_Y[0], s='{:.2f}'.format(res_prob[i,1]),
                      fontsize=FONTSIZE,ha='center',va='center',color='limegreen')
        axs[i,j].text(c[1] / 2 + PLOT_RANGE / 2, TEXT_Y[0], s='{:.2f}'.format(res_prob[i,2]),
                      fontsize=FONTSIZE,ha='center',va='center',color='limegreen')



        ############
        # JPM Part #
        ############

        if j == 0:
            mus_jpm = mus_jpm_s
            sigmas_jpm = sigmas_jpm_s
            res_prob_jpm = res_prob_jpm_s
            ori_prob = np.array(data['AVFus']['synch']['props'][test_index,:,:])
        else:
            mus_jpm = mus_jpm_a
            sigmas_jpm = sigmas_jpm_a
            res_prob_jpm = res_prob_jpm_a
            ori_prob = np.array(data['AVFus']['asynch']['props'][test_index,:,:])

        # mediumorchid,crimson
        axs[i, j].plot(xs, stats.norm.pdf(xs, mus_jpm[i], sigmas_jpm[i]),
                       color='mediumorchid', linewidth=3, label='JPM')
        xs_part = np.linspace(c_jpm[0], c_jpm[1], 100)
        ys_part = stats.norm.pdf(xs_part, mus_jpm[i], sigmas_jpm[i])
        axs[i, j].fill_between(xs_part, ys_part, color='mediumorchid',
                               alpha=ALPHA, label='JPM - DA area',
                               hatch='///')

        axs[i, j].text(c_jpm[0] / 2 - PLOT_RANGE / 2, TEXT_Y[1], s='{:.2f}'.format(res_prob_jpm[i, 0]),
                       fontsize=FONTSIZE, ha='center', va='center', color='mediumorchid')
        axs[i, j].text(c_jpm[0] / 2 + c_jpm[1] / 2, TEXT_Y[1], s='{:.2f}'.format(res_prob_jpm[i, 1]),
                       fontsize=FONTSIZE, ha='center', va='center', color='mediumorchid')
        axs[i, j].text(c_jpm[1] / 2 + PLOT_RANGE / 2, TEXT_Y[1], s='{:.2f}'.format(res_prob_jpm[i, 2]),
                       fontsize=FONTSIZE, ha='center', va='center', color='mediumorchid')


        # raw data (raw probability )
        axs[i, j].text(c_jpm[0] / 2 - PLOT_RANGE / 2, TEXT_Y[2], s='GA\n{}'.format(ori_prob[i, 0]),
                       fontsize=FONTSIZE, ha='center', va='center')
        axs[i, j].text(c_jpm[0] / 2 + c_jpm[1] / 2, TEXT_Y[2], s='DA\n{}'.format(ori_prob[i, 1]),
                       fontsize=FONTSIZE, ha='center', va='center')
        axs[i, j].text(c_jpm[1] / 2 + PLOT_RANGE / 2, TEXT_Y[2], s='BA\n{}'.format(ori_prob[i, 2]),
                       fontsize=FONTSIZE, ha='center', va='center')
        axs[i, j].set_title(sub_plot_title[i])

        #generalsetting
        axs[i, j].set_ylim(0,0.55)


    if i == 0:
        # axs[i, 0].text(-4.75, 0.4, s='p_s={:.4f}'.format(p_s), fontsize=12)
        # axs[i,1].text(-4.75,0.4,s='p_a={:.4f}'.format(p_a),fontsize=12)

        axs[i,1].legend(bbox_to_anchor=(1.04,1),loc="upper left", borderaxespad=0)
        axs[i,0].set_title('Synchrony\n$\sigma_0^2={:.4f}$\nP(C=1)={:.2f}    P(C=2)={:.2f}'.format(sigma_0_s**2,p_s,1-p_s)+'\n\n'+sub_plot_title[i])
        axs[i,1].set_title('Asynchrony\n$\sigma_0^2={:.4f}$\nP(C=1)={:.2f}    P(C=2)={:.2f}'.format(sigma_0_a**2,p_a,1-p_a)+'\n\n'+sub_plot_title[i])


fig.suptitle('tester No.{}'.format(test_index), fontsize=14)
fig.show()


