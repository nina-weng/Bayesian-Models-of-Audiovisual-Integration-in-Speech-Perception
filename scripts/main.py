import pickle
from scipy.optimize import minimize
import numpy as np
import scipy.stats
from scipy.special import factorial
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from utils import get_random_free_params
from fitting_functions import *


# Read in data                 #
pkl_path = '../S2_data/data.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# set the model
model= 'JPM' #'JPM'
implementation='full'
preprocess = True
print('MODEL CHOSE: {}\t{}\tpreprocess:{}'.format(model,implementation,preprocess))


# define the free parameters
init_para = get_random_free_params(model=model,implementation=implementation)

N_trails = 10
num_tester = 16
if model == 'JPM' and implementation == 'full':
    num_params = 14
elif model == 'JPM' and implementation == 'reduced':
    num_params = 13
elif model == 'JPM' and implementation == 'mle':
    num_params = 12
elif model == 'BCI' and implementation == 'full':
    num_params = 14
else:
    raise Exception('model not implemented')


params_stored = np.zeros((N_trails,num_tester,14))
neg_log_record = np.zeros((N_trails,num_tester))

nan_check = [False] * num_tester

if model == 'JPM':
    neg_log_function = neg_log_guassian
elif model == 'BCI':
    neg_log_function = neg_log_bci
else:
    raise Exception('Not Implemented Error')


for j in tqdm(range(num_tester)):
    for i in tqdm(range(N_trails)):
        x0 = get_random_free_params(model=model,implementation=implementation)
        res = minimize(neg_log_function, x0, args=(j, data, model, implementation,preprocess), method='BFGS', tol=1e-4)  # Nelder-Mead,BFGS,L-BFGS-B
        params_stored[i,j,:] = parameter_prepocess(res.x,model=model,implementation=implementation,preprocess=preprocess)

        # record the neg-log value (for choosing the lowest one)
        neg_log = neg_log_function(res.x, j, data, model, implementation,preprocess)
        # neg_log_2 = res.fun
        neg_log_record[i, j] = neg_log

        # if nan_check[j] == False and np.isnan(neg_log) == False:
        #     nan_check[j] = True
        #     i = 0

        print('neg_log for {}th trail & {} tester: {}'.format(i,j,neg_log))
        print(params_stored[i,j,:])


# neg_log_record_n = np.sum(neg_log_record,axis=1)
# print(len(neg_log_record_n))

neg_log_record = np.array(neg_log_record)

min_index = np.nanargmin(neg_log_record,axis=0)
print('min neg_log :{}\ncorresponding index: {}'.format(np.nanmin(neg_log_record,axis=0),min_index))

best_params = []
for i in range(num_tester):
    best_params.append(params_stored[min_index[i],i,:])

best_params = np.array(best_params)
print('best_params:{}'.format(best_params))

# store the best
fitted_param_path = '../fitted_params/fitted_params_jpm_full_4.npy'
np.save(fitted_param_path, best_params)

# store the whole params
fitted_param_path = '../fitted_params/fitted_params_jpm_full_4_all.npy'
np.save(fitted_param_path, params_stored)

# neg_log = []  # use to record the neg-log-multi-nomial likelihood
# def minimize_with_params()
# tester_index = 11
# x0 = np.array(init_para)
# res = minimize(neg_log_guassian, x0, args=(tester_index,data,model,implementation), method='Nelder-Mead', tol=1e-6)  # 'Nelder-Mead' , 'BFGS'

'''
if model == 'JPM' and implementation == 'full':
    [sigma_0_s,sigma_0_a,mu_vg, mu_vb, mu_ag, mu_ab,sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,interval_gd,interval_db] = res.x
elif model == 'JPM' and implementation == 'reduced':
    [sigma_0, mu_vg, mu_vb, mu_ag, mu_ab, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
     interval_gd, interval_db] = res.x
    sigma_0_s,sigma_0_a = 0,sigma_0
else:
    raise Exception('model not implemented')

print('sigma_0:{}\t{}'.format(sigma_0_s,sigma_0_a))
print('mu_vg:{},mu_vb:{},mu_ag:{}, mu_ab:{}'.format(mu_vg, mu_vb, mu_ag, mu_ab))
print('sigma_vh:{}, sigma_vm:{}, sigma_vl:{}, sigma_ah:{}, sigma_am:{}, sigma_al:{}'.format(sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al))
print('interval_gd:{},interval_db:{}'.format(interval_gd,interval_db))
'''


# print(neg_log_guassian(res.x,tester_index,data,model,implementation))

