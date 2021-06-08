import numpy as np
import pickle
from fitting_functions import *
from tqdm import tqdm
from utils import get_random_free_params
from scipy.optimize import minimize

## one trail


def load_data(exp_data_path,param_jpm_path,param_bci_path):
    '''
    load the data from certain path
    :param exp_data_path: experiment data source path
    :param param_jpm_path: fitted JPM model paramters' path
    :param param_bci_path: fitted BCI model paramters' path
    :return: parameters for JPM & BCI, experiment data
    '''
    params_bci = np.load(param_bci_path)
    params_jpm = np.load(param_jpm_path)

    with open(exp_data_path, 'rb') as f:
        exp_data = pickle.load(f)

    return params_jpm,params_bci,exp_data

def load_paramters(params_jpm,params_bci,test_index):
    x0_bci = params_bci[test_index, :]
    p_s, p_a = x0_bci[0:2]
    [mu_vg_bci, mu_vb_bci, mu_ag_bci, mu_ab_bci] = x0_bci[2:6]
    [sigma_vh_bci, sigma_vm_bci, sigma_vl_bci, sigma_ah_bci, sigma_am_bci, sigma_al_bci] = x0_bci[6:12]
    c_bci = x0_bci[12:]

    x0_jpm = params_jpm[test_index, :]
    sigma_0_s, sigma_0_a = x0_jpm[0:2]
    [mu_vg_jpm, mu_vb_jpm, mu_ag_jpm, mu_ab_jpm] = x0_jpm[2:6]
    [sigma_vh_jpm, sigma_vm_jpm, sigma_vl_jpm, sigma_ah_jpm, sigma_am_jpm, sigma_al_jpm] = x0_jpm[6:12]
    c_jpm = x0_jpm[12:]

    # build dictionary to store the data
    params_bci_dict = {'p_s':p_s,'p_a':p_a,
                       'mu_vg':mu_vg_bci,'mu_vb':mu_vb_bci,'mu_ag': mu_ag_bci,'mu_ab': mu_ab_bci,
                       'sigma_vh':sigma_vh_bci,'sigma_vm':sigma_vm_bci,'sigma_vl':sigma_vl_bci,
                       'sigma_ah': sigma_ah_bci,'sigma_am': sigma_am_bci,'sigma_al': sigma_al_bci,
                       'c':c_bci
    }
    params_jpm_dict = {'sigma_0_s': sigma_0_s, 'sigma_0_a': sigma_0_a,
                       'mu_vg': mu_vg_jpm, 'mu_vb': mu_vb_jpm, 'mu_ag': mu_ag_jpm, 'mu_ab': mu_ab_jpm,
                       'sigma_vh': sigma_vh_jpm, 'sigma_vm': sigma_vm_jpm, 'sigma_vl': sigma_vl_jpm,
                       'sigma_ah': sigma_ah_jpm, 'sigma_am': sigma_am_jpm, 'sigma_al': sigma_al_jpm,
                       'c': c_jpm
                       }

    return params_jpm_dict,params_bci_dict


def sample2count(samples, boundaries):
    # extend boundaries
    boundaries = np.insert(boundaries, 0, -np.inf)
    boundaries = np.insert(boundaries, len(boundaries), np.inf)
    #     print(boundaries)

    counts = np.zeros(len(boundaries) - 1)
    for each_s in samples:
        assign2class = 0
        for i in range(len(boundaries) - 1):
            if each_s > boundaries[i] and each_s < boundaries[i + 1]:
                counts[i] += 1
                break
    return counts

def sample_mixture_guassion(mu_1,mu_2,sigma_1,sigma_2,p,sample_size):
    # sample the BCI distribution
    distributions = [
        {"type": np.random.normal, "kwargs": {"loc": mu_1, "scale": sigma_1}},
        {"type": np.random.normal, "kwargs": {"loc": mu_2, "scale": sigma_2}},
    ]
    coefficients = np.array([p, 1 - p])
    coefficients /= coefficients.sum()  # in case these did not add up to 1

    num_distr = len(distributions)
    data_ = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions):
        data_[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
    sample = data_[np.arange(sample_size), random_idx]

    return sample

def generate_samples(params_dict,sample_size_unit,model='JPM'):
    sample_data ={}
    keys = ['AB' ,'AG' ,'VB' ,'VG' ,'AVB' ,'AVG' ,'AVFus']
    for key in keys:
        sample_data[key] = {}

    boundaries = np.linspace(params_dict['mu_ag'], params_dict['mu_ab'],8)

    # generate samples for the single stimuli
    for key in ['AB' ,'AG' ,'VB' ,'VG']:
        mu_key = 'mu_'+key.lower()
        if key[0] == 'A': sigma_key = 'sigma_a'+A_snr[0]
        elif key[0] == 'V': sigma_key = 'sigma_v'+V_snr[0]
        else: raise Exception('Key error.')

        # sigma_keys = ['sigma_'+key.lower()[0] + snr_letter for snr_letter in ['h','m','l']]

        sample_size = sample_size_unit*3
        sample = np.random.normal(params_dict[mu_key], params_dict[sigma_key], sample_size)

        counts = sample2count(sample,boundaries)
        sample_data[key]['counts'] = counts
        sample_data[key]['props'] = list(np.array(counts)/sample_size)

    # generate samples for fusion
    if snr == 'asynch':
        if model == 'JPM':
            for key in ['AVB' ,'AVG' ,'AVFus']:
                if key == 'AVB': mu_key_1,mu_key_2 = 'mu_ab', 'mu_vb'
                elif key == 'AVG': mu_key_1,mu_key_2 = 'mu_ag', 'mu_vg'
                elif key == 'AVFus': mu_key_1,mu_key_2 = 'mu_ab', 'mu_vg'
                else: raise Exception('Key error:{}'.format(key))


                mus_jpm, sigmas_jpm = get_params_AV(params_dict[mu_key_1], params_dict[mu_key_2],
                                    params_dict['sigma_vh'], params_dict['sigma_vm'],params_dict['sigma_vl'],
                                    params_dict['sigma_ah'],params_dict['sigma_am'],params_dict['sigma_al'],
                                    sigma_0=params_dict['sigma_0_a'])

                sample_size =  sample_size_unit*5*2
                sample_jpm = np.random.normal(mus_jpm[fusion_snr_type], sigmas_jpm[fusion_snr_type],sample_size)
                counts = sample2count(sample_jpm,boundaries)
                sample_data[key]['counts'] = counts
                sample_data[key]['props'] = list(np.array(counts) / sample_size)

        elif model == 'BCI':
            for key in ['AVB' ,'AVG' ,'AVFus']:
                if key == 'AVB': mu_key_1,mu_key_2 = 'mu_ab', 'mu_vb'
                elif key == 'AVG': mu_key_1,mu_key_2 = 'mu_ag', 'mu_vg'
                elif key == 'AVFus': mu_key_1,mu_key_2 = 'mu_ab', 'mu_vg'
                else: raise Exception('Key error:{}'.format(key))

                mus_pc1, sigmas_pc1 = get_params_AV(params_dict[mu_key_1], params_dict[mu_key_2],
                                    params_dict['sigma_vh'], params_dict['sigma_vm'],params_dict['sigma_vl'],
                                    params_dict['sigma_ah'],params_dict['sigma_am'],params_dict['sigma_al'],
                                    sigma_0=0)
                mus_pc2, sigmas_pc2 = get_params_AV(params_dict[mu_key_1], params_dict[mu_key_2],
                                    params_dict['sigma_vh'], params_dict['sigma_vm'],params_dict['sigma_vl'],
                                    params_dict['sigma_ah'],params_dict['sigma_am'],params_dict['sigma_al'],
                                    sigma_0=np.inf)

                sample_size = sample_size_unit * 5 * 2
                sample_bci = sample_mixture_guassion(mus_pc1[fusion_snr_type], mus_pc2[fusion_snr_type],
                            sigmas_pc1[fusion_snr_type], sigmas_pc2[fusion_snr_type], params_dict['p_a'],
                                                     sample_size)
                counts = sample2count(sample_bci, boundaries)
                sample_data[key]['counts'] = counts
                sample_data[key]['props'] = list(np.array(counts) / sample_size)



        else: raise Exception('Not implemented model = {}'.format(model))
    else:
        raise Exception('Not implemented snr = {}'.format(snr))



    return sample_data


def fit_model(tester_index,exp_data, N_trails,model = 'JPM', implementation='full',preprocess=True):
    if model == 'JPM':
        neg_log_function = neg_log_jpm_for1tester
    elif model == 'BCI':
        neg_log_function = neg_log_bci_for1tester
    else:
        raise Exception('Not Implemented model:{}'.format(model))

    if implementation == 'full':num_parameters=20
    # elif implementation =='reduced': num_parameters=
    # elif implementation == 'mle': num_parameters=
    else: raise Exception('Not implemented implementation method:{}'.format(implementation))

    params_stored = np.zeros((N_trails,num_parameters))
    neg_log_record = np.zeros(N_trails)

    for i in tqdm(range(N_trails)):
        # get free parameters first
        x0 = np.random.rand(num_parameters)
        res = minimize(neg_log_function, x0, args=(exp_data, model, implementation, preprocess), method='BFGS',
                       tol=1e-4)  # Nelder-Mead,BFGS,L-BFGS-B
        params_stored[i, :] = parameter_prepocess_9cates(res.x, model=model, implementation=implementation,
                                                     preprocess=preprocess)

        # record the neg-log value (for choosing the lowest one)
        neg_log = neg_log_function(res.x, exp_data, model, implementation, preprocess)
        neg_log_record[i] = neg_log


        print('neg_log for {}th trail & {} tester: {}'.format(i, tester_index, neg_log))

    min_index = np.nanargmin(neg_log_record)
    print('min neg_log :{}\ncorresponding index: {}'.format(np.nanmin(neg_log_record), min_index))
    return np.nanmin(neg_log_record)



# define the parameters
tester_number = 7
V_snr = 'high'
A_snr = 'low'
snr = 'asynch'  # 'synch'
sample_size_unit = 25

if V_snr == 'high' and A_snr =='high': fusion_snr_type = 0
elif V_snr == 'mid' and A_snr =='high':fusion_snr_type = 1
elif V_snr == 'low' and A_snr =='high':fusion_snr_type = 2
elif V_snr == 'high' and A_snr =='mid':fusion_snr_type = 3
elif V_snr == 'high' and A_snr =='low':fusion_snr_type = 4
else: raise Exception('fusion snr not implemented.')


# load in the data
param_bci_path = '../fitted_params/fitted_params_bci_full_4.npy'
param_jpm_path = '../fitted_params/fitted_params_jpm_full_4.npy'
print('jpm parameters from :{}'.format(param_jpm_path))
print('bci parameters from :{}'.format(param_bci_path))
pkl_path = '../S2_data/data.pkl'
print('experiment data from :{}'.format(pkl_path))

params_jpm,params_bci,exp_data = load_data(pkl_path,param_jpm_path,param_bci_path)

# get the parameters for chosen tester
params_jpm_dict,params_bci_dict = load_paramters(params_jpm,params_bci,tester_number)

# generating samples (1 trail)
# data_sample = generate_samples(params_jpm_dict,sample_size_unit,model='JPM')
data_sample = generate_samples(params_bci_dict,sample_size_unit,model='BCI')
print(data_sample)

# fit the sample with JPM and BCI model
N_trails = 5
jpm_neg_log_sum = fit_model(tester_number,data_sample, N_trails,model = 'JPM', implementation='full',preprocess=True)
bci_neg_log_sum = fit_model(tester_number,data_sample, N_trails,model = 'BCI', implementation='full',preprocess=True)
print('jpm_neg_log_sum:{:.4f}'.format(jpm_neg_log_sum))
print('bci_neg_log_sum:{:.4f}'.format(bci_neg_log_sum))
