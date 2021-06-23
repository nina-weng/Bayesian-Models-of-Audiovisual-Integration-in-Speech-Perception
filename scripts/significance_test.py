"""
significance test 16 testers
H_a_0 : for JPM samples, s_JPM > s_BCI
H_b_0 : for BCI samples, s_JPM < s_BCI
Notice that null hypothesis is something we are trying to prove against
"""

import os
import numpy as np
from sample_fitted_result_analyse import load_result_data_v2
from scipy import stats
import matplotlib.pyplot as plt


def plot_distribution(data1,data2,bins,color1,color2,title):
    plt.figure(figsize=(6,6))
    plt.hist(data1,bins=bins,color=color1,alpha=0.5,label='JPM sample')
    plt.hist(data2,bins=bins,color=color2,alpha=0.5,label='BCI sample')
    plt.vlines(0,0,9,color='grey',linestyles='dashed')
    plt.xlabel('diff(s_JPM-s_BCI)')
    plt.ylabel('count')
    plt.ylim(0,9)
    plt.title(title)
    plt.legend()
    plt.show()



if __name__ == '__main__':

    # get the sampling files
    fitted_result_dir = '../results/fitted_sampling'
    f_list = os.listdir(fitted_result_dir)

    num_of_tester = 16
    N_experiment = 100
    h_boundaries = 0
    alpha = 0.01

    # s_index: type of score
    # 0: only fusion neg-log-likelihood; 1: sum of all conditions' neg-log-likelihood; 2: 1 + regularization term
    s_index = 0

    data_jpm_sample = np.zeros((num_of_tester,N_experiment))
    data_bci_sample = np.zeros((num_of_tester,N_experiment))

    if len(f_list)!= num_of_tester*2:
        print('WARNING: file number not match!')

    for fname in f_list:
        print('processing file:{}'.format(fname))
        test_id = int(fname.split('_')[0])
        sample_type = fname.split('_')[1]
        fpath = os.path.join(fitted_result_dir,fname)
        if sample_type == 'jpm':
            jpm_scores, bci_scores, inf_count = load_result_data_v2(fpath)
            # data_jpm_sample[test_id, :] = jpm_scores[:, s_index] - bci_scores[:, s_index]
            data_jpm_sample[test_id,:] = (jpm_scores[:,s_index]-bci_scores[:,s_index])/bci_scores[:,s_index]
        elif sample_type == 'bci':
            jpm_scores, bci_scores, inf_count = load_result_data_v2(fpath)
            # data_bci_sample[test_id, :] = jpm_scores[:, s_index] - bci_scores[:, s_index]
            data_bci_sample[test_id,:] = (jpm_scores[:,s_index]-bci_scores[:,s_index])/bci_scores[:,s_index]

    print('test')
    # calculate the p-value
    p_vals_jpm = np.zeros(N_experiment)
    p_vals_bci = np.zeros(N_experiment)

    for j in range(N_experiment):
        #JPM samples
        statistic,pvalue = stats.ttest_1samp(data_jpm_sample[:,j],popmean=h_boundaries,alternative='less')
        # print(np.mean(data_jpm_sample[:,j]))
        # print('{}\t{}\t{}'.format(j,statistic,pvalue))
        p_vals_jpm[j] = pvalue
        #BCI samples
        statistic, pvalue = stats.ttest_1samp(data_bci_sample[:, j], popmean=h_boundaries, alternative='greater')
        # print(np.mean(data_bci_sample[:, j]))
        # print('{}\t{}\t{}'.format(j, statistic, pvalue))
        p_vals_bci[j] = pvalue

        if j <3:
            # print(np.arange(-10,11,0.5))
            plot_distribution(data1=data_jpm_sample[:,j],data2=data_bci_sample[:,j],bins=np.arange(-1,1,0.05),
                              color1='violet',color2='seagreen',title='s_index:{} alpha:{}'.format(s_index,alpha))



    print('H_a_0, p-value less then {}:{}'.format(alpha,np.sum(p_vals_jpm<alpha)))
    print('H_b_0, p-value less then {}:{}'.format(alpha, np.sum(p_vals_bci < alpha)))



