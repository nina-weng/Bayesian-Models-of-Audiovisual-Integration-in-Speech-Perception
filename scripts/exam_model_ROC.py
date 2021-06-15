from exam_models import *

if __name__ == '__main__':
    # define the parameters
    tester_number = 12
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

        print(data_sample_bci['AVFus']['counts'])

