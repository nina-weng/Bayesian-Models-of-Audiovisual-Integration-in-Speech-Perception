from exam_models import *
from scipy.stats import norm

def plot_ROC(p_fa,p_hit,title_info):
    plt.figure(figsize = (6,6))
    p_fa = np.concatenate([[0],p_fa])
    p_hit = np.concatenate([[0],p_hit])
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
    N_experiment = 1

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

    # get the parameters for chosen tester
    params_jpm_dict, params_bci_dict = load_paramters(params_jpm, params_bci, tester_number)

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

        plot_ROC(avf_p_fa,avf_p_hit,title_info='tester id :{}'.format(tester_number))
        plot_ROC_gaussian_coordinates(avf_p_fa,avf_p_hit,title_info='tester id :{}'.format(tester_number))



