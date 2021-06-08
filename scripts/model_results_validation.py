import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


WIDTH = 0.2

fitted_param_path_jpm = '../fitted_params/fitted_params_jpm_full_4.npy'
fitted_param_path_bci = '../fitted_params/fitted_params_bci_full_4.npy'
params_stored_jpm= np.load(fitted_param_path_jpm)
params_stored_bci= np.load(fitted_param_path_bci)

# print(np.mean(params_stored_jpm,axis=0))
# print(np.std(params_stored_jpm,axis=0)/16)
#
# print(np.mean(params_stored_bci,axis=0))
# print(np.std(params_stored_bci,axis=0)/16)


# Fig 4A
fig, axs = plt.subplots(1,2, figsize=(6, 6))

sigma_0_s = params_stored_jpm[:,0]
sigma_0_a = params_stored_jpm[:,1]
# print(sigma_0_s)
# print(sigma_0_s.shape)

p_s = params_stored_bci[:,0]
p_a = params_stored_bci[:,1]

sigma_mean = np.array([np.mean(sigma_0_s**2),np.mean(sigma_0_a**2)])
sigma_errs = np.array([np.std(sigma_0_s**2)/np.sqrt(16),np.std(sigma_0_a**2)/np.sqrt(16)])

p_mean = np.array([np.mean(1-p_s),np.mean(1-p_a)])
p_errs = np.array([np.std(1-p_s)/np.sqrt(16),np.std(1-p_a)/np.sqrt(16)])


labels = ['synch','asynch']
bar_color = ['steelblue','tomato']
pos = [0.3,0.7]
titles = ['Joint Prior:$\sigma_0^2$','BCI:P(C=2)']

for i in range(2):
    axs[0].bar(pos[i], sigma_mean[i], color =bar_color[i],
                 alpha=0.9,yerr = sigma_errs[i],width=WIDTH,
               capsize=5,ecolor='k',edgecolor='k')
    axs[1].bar(pos[i],p_mean[i],color =bar_color[i],
                 alpha=0.9,yerr = p_errs[i],width=WIDTH,
               capsize=5,ecolor='k',edgecolor='k')

    axs[i].set_xticks(pos)
    axs[i].set_xticklabels(labels)
    axs[i].set_title(titles[i])

# text - one sample one sided t-test
t_jpm = stats.ttest_1samp(sigma_0_a**2,popmean=0,alternative = 'greater')
t_bci = stats.ttest_1samp(1-p_a,popmean=0,alternative = 'greater')
ts = np.array([t_jpm.pvalue,t_bci.pvalue])
axs[0].text(pos[1],sigma_mean[1]+sigma_errs[1]+0.02,s='p={:.4f}'.format(ts[0]),ha='center')
axs[1].text(pos[1],p_mean[1]+p_errs[1]+0.02,s='p={:.4f}'.format(ts[1]),ha='center')

# test - two sample one sided t-test
t_jpm_two = stats.ttest_ind(sigma_0_s**2,sigma_0_a**2,alternative='less')
t_bci_two = stats.ttest_ind(1-p_s,1-p_a,alternative='less')
ts_two = np.array([t_jpm_two.pvalue,t_bci_two.pvalue])
tmp_height = np.array([sigma_mean[1]+sigma_errs[1]+0.02,p_mean[1]+p_errs[1]+0.02])
for i in range(2):
    # axs[i].axline((pos[0],tmp_height[i]+0.1),(pos[1],tmp_height[i]+0.1))
    axs[i].plot([pos[0],pos[1]],[tmp_height[i]+0.1,tmp_height[i]+0.1],color='k')
    axs[i].plot([pos[0], pos[0]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    axs[i].plot([pos[1], pos[1]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    axs[i].text(pos[0]/2+pos[1]/2, tmp_height[i] + 0.15, s='p={:.4g}'.format(ts_two[i]), ha='center')

axs[0].set_ylim(0,1.5)
axs[1].set_ylim(0,1.5)

fig.suptitle('Prior params')
fig.show()


# FIG B: visual reliability
plt.figure(figsize=(6,6))
sigma_v_jpm = params_stored_jpm[:,6:9]
sigma_a_jpm = params_stored_jpm[:,9:12]
sigma_v_bci = params_stored_bci[:,6:9]
sigma_a_bci = params_stored_bci[:,9:12]

labels = ['high','mid','low']
x = np.arange(len(labels))
width = 0.3
sigma_v_jpm_mean = np.mean(1/sigma_v_jpm**2,axis=0)
sigma_a_jpm_mean = np.mean(1/sigma_a_jpm**2,axis=0)

sigma_v_jpm_err = np.std(1/sigma_v_jpm**2,axis=0)/np.sqrt(16)
sigma_a_jpm_err = np.std(1/sigma_a_jpm**2,axis=0)/np.sqrt(16)

sigma_v_bci_mean = np.mean(1/sigma_v_bci**2,axis=0)
sigma_a_bci_mean = np.mean(1/sigma_a_bci**2,axis=0)

sigma_v_bci_err = np.std(1/sigma_v_bci**2,axis=0)/np.sqrt(16)
sigma_a_bci_err = np.std(1/sigma_a_bci**2,axis=0)/np.sqrt(16)

# print(sigma_v_jpm_mean.shape)
# # print(sigma_v_jpm_mean)

rects1 = plt.bar(x - width/2, sigma_v_jpm_mean, color = 'steelblue',
                 alpha=0.9,yerr = sigma_v_jpm_err,width=7*width/8,
                 label='joint prior',capsize=5,ecolor='k',edgecolor='k')
rects2 = plt.bar(x + width/2, sigma_v_bci_mean,color = 'skyblue',
                 alpha=0.9,yerr =sigma_v_bci_err, width=7*width/8,
                 label='BCI',capsize=5,ecolor='k',edgecolor='k')

# t-test
sigma_vhigh = np.concatenate((sigma_v_jpm[:,0],sigma_v_bci[:,0]),axis = 0)
sigma_vmid = np.concatenate((sigma_v_jpm[:,1],sigma_v_bci[:,1]),axis = 0)
sigma_vlow = np.concatenate((sigma_v_jpm[:,2],sigma_v_bci[:,2]),axis = 0)

res_high_mid_v = stats.wilcoxon(sigma_vhigh,sigma_vmid,alternative='less')
res_mid_low_v = stats.wilcoxon(sigma_vmid,sigma_vlow,alternative='less')

tmp_height = np.array([sigma_v_jpm_mean[0]+sigma_v_jpm_err[0],
                       sigma_v_jpm_mean[1]+sigma_v_jpm_err[1]])

res_wilconxon = np.array([res_high_mid_v.pvalue,res_mid_low_v.pvalue])

for i in range(2):
    plt.plot([x[i],x[i+1]],[tmp_height[i]+0.1,tmp_height[i]+0.1],color='k')
    plt.plot([x[i], x[i]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i+1], x[i+1]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i]-0.1, x[i]+0.1], [tmp_height[i] + 0.07, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i+1]-0.1, x[i+1]+0.1], [tmp_height[i] + 0.07, tmp_height[i] + 0.07], color='k')
    plt.text(x[i]/2+x[i+1]/2, tmp_height[i] + 0.15, s='p={:.4g}'.format(res_wilconxon[i]), ha='center')



plt.xticks(x,labels)
# plt.xticklabels(labels)
plt.legend()
plt.title('Visual reliability\n$1/\sigma_v^2$')
plt.ylim(0,1.5)
plt.show()


sigma_ahigh = np.concatenate((sigma_a_jpm[:,0],sigma_a_bci[:,0]),axis = 0)
sigma_amid = np.concatenate((sigma_a_jpm[:,1],sigma_a_bci[:,1]),axis = 0)
sigma_alow = np.concatenate((sigma_a_jpm[:,2],sigma_a_bci[:,2]),axis = 0)
#
# print(sigma_vlow.shape)

res_high_mid_a = stats.wilcoxon(sigma_ahigh,sigma_amid,alternative='less')
res_mid_low_a = stats.wilcoxon(sigma_amid,sigma_alow,alternative='less')

plt.figure(figsize=(6,6))
rects1 = plt.bar(x - width/2, sigma_a_jpm_mean, color = 'steelblue',
                 alpha=0.9,yerr = sigma_a_jpm_err,width=7*width/8,
                 label='joint prior',capsize=5,ecolor='k',edgecolor='k')
rects2 = plt.bar(x + width/2, sigma_a_bci_mean,color = 'skyblue',
                 alpha=0.9,yerr =sigma_a_bci_err, width=7*width/8,
                 label='BCI',capsize=5,ecolor='k',edgecolor='k')


res_wilconxon = np.array([res_high_mid_a.pvalue,res_mid_low_a.pvalue])

tmp_height = np.array([sigma_a_jpm_mean[0]+sigma_a_jpm_err[0],
                       sigma_a_jpm_mean[1]+sigma_a_jpm_err[1]])

for i in range(2):
    plt.plot([x[i],x[i+1]],[tmp_height[i]+0.1,tmp_height[i]+0.1],color='k')
    plt.plot([x[i], x[i]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i+1], x[i+1]], [tmp_height[i] + 0.1, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i]-0.1, x[i]+0.1], [tmp_height[i] + 0.07, tmp_height[i] + 0.07], color='k')
    plt.plot([x[i+1]-0.1, x[i+1]+0.1], [tmp_height[i] + 0.07, tmp_height[i] + 0.07], color='k')
    plt.text(x[i]/2+x[i+1]/2, tmp_height[i] + 0.15, s='p={:.4g}'.format(res_wilconxon[i]), ha='center')



plt.xticks(x,labels)
# plt.xticklabels(labels)
plt.legend()
plt.title('Auditory reliability\n$1/\sigma_a^2$')
plt.ylim(0,1.5)
plt.show()



# FIG 4 - E
plt.figure(figsize=(6,6))

