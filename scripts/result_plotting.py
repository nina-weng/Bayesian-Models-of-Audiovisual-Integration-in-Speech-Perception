import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
from fitting_functions import *

# Read in data                 #
pkl_path = '../S2_data/data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# get the params
# fitted_param_path = '../S2_data/fitted_params_6.npy'
# fitted_param_path = '../fitted_params/fitted_params_11.npy'
# fitted_param_path = '../fitted_params/fitted_params_bci_full_1.npy'
fitted_param_path = '../fitted_params/fitted_params_bci_full_4.npy'
params_stored= np.load(fitted_param_path)
print(params_stored.shape)
# print(params_stored)
print(np.mean(params_stored,axis=0))
print(np.std(params_stored,axis=0)/4)

model = 'JPM'
implementation ='full'

# verify
neg_log_sum = 0
for i in range(16):
    if model == 'JPM':
        neg_log = neg_log_guassian(params_stored[i,:], i, data, model=model, implementation=implementation,preprocess=False)
    # params_stored[i,:] = parameter_prepocess(params_stored[i,:],model='JPM',implementation='full',preprocess=True)
    elif model =='BCI':
        neg_log =neg_log_bci(params_stored[i,:], i, data, model=model, implementation=implementation, preprocess = False)
    else:
        raise Exception('Not Implemented Error.')

    # print(params_stored[i,:])
    print('{}:{}'.format(i,neg_log))
    neg_log_sum+=neg_log
print('########\nneg log sum:{}\n########'.format(neg_log_sum))

num_tester = 16
props_with_fitted_data = {}
keys = ['AB' ,'AG' ,'VB' ,'VG' ,'AVB' ,'AVG' ,'AVFus','AVB-A' ,'AVG-A' ,'AVFus-A']
mu_index = {'AB' :5 ,'AG' :4 ,'VB' :3 ,'VG' :2}

for key in keys:
    if len(key) ==2:
        props_with_fitted_data[key] = np.zeros((num_tester ,3 ,3))
    else:
        props_with_fitted_data[key] = np.zeros((num_tester ,5 ,3))
    for i in range(num_tester):
        [sigma_0_s, sigma_0_a ,mu_vg, mu_vb, mu_ag, mu_ab ,sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al ,interval_gd
         ,interval_db] = params_stored[i ,:]

        if len(key) ==2:
            mu = params_stored[i ,mu_index[key]]
            mus = np.array([mu ] *3)
            if key[0] == 'A':
                sigmas = np.array(params_stored[i ,9:12])
            elif key[0] == 'V':
                sigmas = np.array(params_stored[i, 6:9])
        else:
            if key in ['AVG','AVB','AVFus']:
                sigma_0 = sigma_0_s
            else: sigma_0 = sigma_0_a

            # for AVB AVG AVFus
            if key[2] == 'B':
                mus ,sigmas = get_params_AV(mu_ab ,mu_vb ,sigma_vh ,sigma_vm ,sigma_vl ,sigma_ah ,sigma_am ,sigma_al,sigma_0= sigma_0)
            elif key[2] == 'G':
                mus ,sigmas = get_params_AV(mu_ag ,mu_vg ,sigma_vh ,sigma_vm ,sigma_vl ,sigma_ah ,sigma_am ,sigma_al,sigma_0= sigma_0)
            elif key[2] == 'F':
                mus ,sigmas = get_params_AV(mu_ab ,mu_vg ,sigma_vh ,sigma_vm ,sigma_vl ,sigma_ah ,sigma_am ,sigma_al,sigma_0= sigma_0)
            else:
                print('WARNING!')
        c = params_stored[i ,-2:]
        props = guassian2prob(mus ,sigmas ,c)
        #         print(props_with_fitted_data)
        props_with_fitted_data[key][i ,: ,:] = props

########################################
# PLOT the signal stimuli ones
########################################

snrs_index = ['high','mid','low']
fig,ax= plt.subplots(3,4,figsize=(12,12))

ks = ['AB','AG','VB','VG']

for j,k in enumerate(ks):

    labels = ['G', 'D', 'B']
    props_raw = np.nanmean(data[k]['props'],axis = 0)
    props_fitted = np.nanmean(props_with_fitted_data[k],axis=0)
    err_raw = np.nanstd(data[k]['props'],axis = 0)/np.sqrt(num_tester)
    err_fitted = np.nanstd(props_with_fitted_data[k],axis = 0)/np.sqrt(num_tester)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    for i in range(3):
        rects1 = ax[i,j].bar(x - width/2, props_raw[i], color = 'steelblue',alpha=0.9,yerr = err_raw[i],width=width, label='observation',capsize=5,ecolor='red')
        rects2 = ax[i,j].bar(x + width/2, props_fitted[i],color = 'skyblue',alpha=0.9,yerr = err_fitted[i], width=width, label='prediction',capsize=5,ecolor='red')
        ax[i,j].set_xticks(x)
        ax[i,j].set_xticklabels(labels)
        if (i==0 and j==3):
            ax[i,j].legend()
        ax[i,j].bar_label(rects1, padding=3,fmt='%.2f')
        ax[i,j].bar_label(rects2, padding=3,fmt='%.2f')
        ax[i,j].set_ylim(0,1)
        ax[i,j].set_title(k+'-'+snrs_index[i])

plt.show()


########################################
# PLOT the audiovisual stimuli         #
########################################

def plot_avsituation(k):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    #     k = 'AVB'

    labels = ['G', 'D', 'B']
    props_raw = np.nanmean(data[k]['synch']['props'], axis=0)
    props_fitted = np.nanmean(props_with_fitted_data[k], axis=0)
    err_raw = np.nanstd(data[k]['synch']['props'], axis=0) / np.sqrt(num_tester)
    err_fitted = np.nanstd(props_with_fitted_data[k], axis=0) / np.sqrt(num_tester)

    props_raw_a = np.nanmean(data[k]['asynch']['props'], axis=0)
    props_fitted_a = np.nanmean(props_with_fitted_data[k+'-A'], axis=0)
    err_raw_a = np.nanstd(data[k]['asynch']['props'], axis=0) / np.sqrt(num_tester)
    err_fitted_a = np.nanstd(props_with_fitted_data[k+'-A'], axis=0) / np.sqrt(num_tester)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    cnt = 0
    for i in range(3):
        for j in range(3):
            if i > 0 and j > 0:
                continue

            rects1 = ax[i, j].bar(x - 5*width / 8, props_raw[cnt], color='steelblue', alpha=0.9, yerr=err_raw[cnt],
                                  width=width/4, label='observation', capsize=5, ecolor='red')
            rects2 = ax[i, j].bar(x - 3*width / 8, props_fitted[cnt], color='skyblue', alpha=0.9, yerr=err_fitted[cnt],
                                  width=width/4, label='prediction', capsize=5, ecolor='red')

            rects3 = ax[i, j].bar(x + 3*width / 8, props_raw_a[cnt], color='darkorange', alpha=0.9, yerr=err_raw_a[cnt],
                                  width=width/4, label='observation', capsize=5, ecolor='red')
            rects4 = ax[i, j].bar(x + 5*width / 8, props_fitted_a[cnt], color='orange', alpha=0.4, yerr=err_fitted_a[cnt],
                                  width=width/4, label='prediction', capsize=5, ecolor='red')


            ax[i, j].set_xticks(x)
            ax[i, j].set_xticklabels(labels)
            if (i == 0 and j == 2):
                ax[i, j].legend()
            ax[i, j].bar_label(rects1, padding=3, fmt='%.2f')
            ax[i, j].bar_label(rects2, padding=3, fmt='%.2f')
            ax[i, j].bar_label(rects3, padding=3, fmt='%.2f')
            ax[i, j].bar_label(rects4, padding=3, fmt='%.2f')

            ax[i, j].set_ylim(0, 1)
            ax[i, j].set_title(k + '  ' + 'V-' + snrs_index[j] + ',A-' + snrs_index[i]+'[{},{}]'.format(i,j))
            cnt += 1

    plt.show()


plot_avsituation('AVB')
plot_avsituation('AVG')
plot_avsituation('AVFus')
