from exam_models import *
from scipy.stats import norm




def plot_ROC(p_fa,p_hit,title_info,error_p_fa=None,error_p_hit=None):
    plt.figure(figsize = (6,6))
    # p_fa = np.concatenate([[0],p_fa])
    # p_hit = np.concatenate([[0],p_hit])
    if error_p_fa.all()!=None and error_p_hit.all()!=None:
        print('error bar plot')
        plt.errorbar(p_fa, p_hit, xerr= error_p_fa,yerr=error_p_hit,
                     marker='o',markerfacecolor='None', color='k', markersize=1,
                     ecolor = 'red')
    else:
        plt.plot(p_fa,p_hit,marker = 'o',markerfacecolor= 'red',color='k',markersize=10)
    xs=np.linspace(0,1,100)
    plt.plot(xs,xs,color='grey',alpha=0.5)
    plt.title('ROC - {}'.format(title_info))
    plt.xlabel('$P_{FA}$')
    plt.ylabel('$P_{HIT}$')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

    return None


def plot_ROC_gaussian_coordinates(p_fa,p_hit,title_info):
    plt.figure(figsize = (6,6))
    inv_p_hit = [norm.ppf(p_h) for p_h in p_hit]
    inv_p_fa = [norm.ppf(p_f) for p_f in p_fa]
    plt.plot(inv_p_fa,inv_p_hit,marker = 'o',color= 'orangered',markersize=8,linestyle="None")
    xs=np.linspace(-2,2,100)
    xs_ = np.zeros(100)
    plt.plot(xs_,xs,color='grey',alpha=0.5)
    plt.plot(xs, xs_, color='grey', alpha=0.5)
    plt.title('ROC in Gaussian Coordinates - {}'.format(title_info))
    plt.xlabel('$Z_{FA}$')
    plt.ylabel('$Z_{HIT}$')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

    return None




if __name__ == '__main__':
    # define the parameters
    tester_number = 1
    V_snr = 'high'
    A_snr = 'low'
    snr = 'asynch'  # 'synch'
    sample_size_unit = 25
    N_experiment = 100

    if V_snr == 'high' and A_snr == 'high':
        fusion_snr_type = 0
    elif V_snr == 'mid' and A_snr == 'high':
        fusion_snr_type = 1
    elif V_snr == 'low' and A_snr == 'high':
        fusion_snr_type = 2
    elif V_snr == 'high' and A_snr == 'mid':
        fusion_snr_type = 3
    elif V_snr == 'high' and A_snr == 'low':
        fusion_snr_type = 4
    else:
        raise Exception('fusion snr not implemented.')

    # load in the data
    param_bci_path = '../fitted_params/fitted_params_bci_full_4.npy'
    param_jpm_path = '../fitted_params/fitted_params_jpm_full_4.npy'
    print('jpm parameters from :{}'.format(param_jpm_path))
    print('bci parameters from :{}'.format(param_bci_path))
    pkl_path = '../S2_data/data.pkl'
    print('experiment data from :{}'.format(pkl_path))

    params_jpm, params_bci, exp_data = load_data(pkl_path, param_jpm_path, param_bci_path)

    subplot_row = 4
    subplot_col = 4
    fig,axes = plt.subplots(subplot_row,subplot_col, figsize=(12, 12))

    fig_gau, axes_gau = plt.subplots(subplot_row, subplot_col, figsize=(12, 12))

    # get the parameters for all tester
    for i in tqdm(range(16)):
        # get axes index
        axes_idx_row = int(math.floor(i/subplot_col))
        axes_idx_col = i%subplot_col

        # print(axes_idx_row,axes_idx_col)


        # get tester id
        tester_number = i

        # get parameters for chosen tester id
        params_jpm_dict, params_bci_dict = load_paramters(params_jpm, params_bci, tester_number)

        p_fa_collects = []
        p_hit_collects = []

        for i_exp in range(N_experiment):
            # generating samples (1 trail)
            data_sample_jpm = generate_samples(params_jpm_dict,sample_size_unit,A_snr=A_snr, V_snr=V_snr,snr=snr,\
                                           fusion_snr_type=fusion_snr_type,model='JPM')
            data_sample_bci = generate_samples(params_bci_dict,sample_size_unit,A_snr=A_snr, V_snr=V_snr,snr=snr,\
                                           fusion_snr_type=fusion_snr_type,model='BCI')

            avf_prop_jpm = data_sample_jpm['AVFus']['props']
            avf_prop_bci = data_sample_bci['AVFus']['props']

            avf_p_fa = np.cumsum(avf_prop_jpm[::-1])
            avf_p_hit = np.cumsum(avf_prop_bci[::-1])

            p_fa = np.concatenate([[0], avf_p_fa])
            p_hit = np.concatenate([[0], avf_p_hit])

            p_fa_collects.append(p_fa)
            p_hit_collects.append(p_hit)

            # plot_ROC(avf_p_fa,avf_p_hit,title_info='tester id :{}'.format(tester_number))
            # plot_ROC_gaussian_coordinates(avf_p_fa,avf_p_hit,title_info='tester id :{}'.format(tester_number))


        # convert to numpy
        p_hit_collects = np.array(p_hit_collects)
        p_fa_collects = np.array(p_fa_collects)

        # get error bar
        error_p_fa = np.std(p_fa_collects,axis=0) #/np.sqrt(N_experiment)
        error_p_hit = np.std(p_hit_collects, axis=0) #/np.sqrt(N_experiment)

        mean_p_fa = np.mean(p_fa_collects,axis=0)
        mean_p_hit = np.mean(p_hit_collects,axis=0)

        # plot_ROC(mean_p_fa, mean_p_hit, title_info='tester id:{}  number of experiment:{}'.format(tester_number,N_experiment),error_p_fa=error_p_fa,error_p_hit=error_p_hit)
        axes[axes_idx_row,axes_idx_col].errorbar(mean_p_fa, mean_p_hit, xerr= error_p_fa,yerr=error_p_hit,
                     marker='o',markerfacecolor='None', color='k', markersize=1,
                     ecolor = 'red',elinewidth = 1)
        xs = np.linspace(0, 1, 100)
        axes[axes_idx_row, axes_idx_col].plot(xs, xs, color='grey', alpha=0.5)
        axes[axes_idx_row, axes_idx_col].set_title('tester id:{}'.format(tester_number))
        axes[axes_idx_row, axes_idx_col].set_xlabel('$P_{FA}$')
        axes[axes_idx_row, axes_idx_col].set_ylabel('$P_{HIT}$')
        axes[axes_idx_row, axes_idx_col].set_xlim(0, 1)
        axes[axes_idx_row, axes_idx_col].set_ylim(0, 1)


        # roc in gaussian corrdinate
        inv_p_hit_collects = norm.ppf(p_hit_collects)
        inv_p_fa_collects = norm.ppf(p_fa_collects)

        error_inv_p_fa = np.nanstd(inv_p_fa_collects, axis=0)  # /np.sqrt(N_experiment)
        error_inv_p_hit = np.nanstd(inv_p_hit_collects, axis=0)  # /np.sqrt(N_experiment)

        mean_inv_p_fa = np.nanmean(inv_p_fa_collects, axis=0)
        mean_inv_p_hit = np.nanmean(inv_p_hit_collects, axis=0)

        axes_gau[axes_idx_row, axes_idx_col].errorbar(mean_inv_p_fa, mean_inv_p_hit,
                     xerr= error_inv_p_fa,yerr=error_inv_p_hit,
                     ecolor = 'black',elinewidth = 1,linestyle="None")
        axes_gau[axes_idx_row,axes_idx_col].scatter(mean_inv_p_fa, mean_inv_p_hit,marker='o', color='royalblue', s=20,alpha=0.5)
        xs = np.linspace(-2, 2, 100)
        xs_ = np.zeros(100)
        axes_gau[axes_idx_row, axes_idx_col].plot(xs_, xs, color='grey', alpha=0.5)
        axes_gau[axes_idx_row, axes_idx_col].plot(xs, xs_, color='grey', alpha=0.5)
        axes_gau[axes_idx_row, axes_idx_col].set_title('tester id:{}'.format(tester_number))
        axes_gau[axes_idx_row, axes_idx_col].set_xlabel('$Z_{FA}$')
        axes_gau[axes_idx_row, axes_idx_col].set_ylabel('$Z_{HIT}$')
        axes_gau[axes_idx_row, axes_idx_col].set_xlim(-2, 2)
        axes_gau[axes_idx_row, axes_idx_col].set_ylim(-2, 2)



    fig.suptitle('ROC curve\n(number of experiment:{}, error bar:{})'.format(N_experiment,'standard deviation'),fontsize=18)
    fig.show()

    fig_gau.suptitle('ROC in Gaussian Coordinates\n(number of experiment:{}, error bar:{})'.format(N_experiment, 'standard deviation'),
                 fontsize=18)
    fig_gau.show()
