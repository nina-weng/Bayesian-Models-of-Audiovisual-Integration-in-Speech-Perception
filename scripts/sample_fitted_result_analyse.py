import matplotlib.pyplot as plt
import numpy as np
import math
import os

COLOR_BCI = 'limegreen'
COLOR_JPM = 'mediumorchid'


def load_result_data(txt_path):
    jpm_scores = []
    bci_scores = []
    inf_count = 0
    with open(txt_path, "r") as f:
        for idx,line in enumerate(f.readlines()):
            if idx == 0 or idx == 1:
                continue
            contents = line.split('\t')[1]
            jpm_s = float(contents.split(',')[0].split(':')[1])
            bci_s = float(contents.split(',')[1].split(':')[1][:-2])
            # print(jpm_s,bci_s)

            if jpm_s == np.inf or bci_s == np.inf:
                inf_count += 1
                continue

            jpm_scores.append(jpm_s)
            bci_scores.append(bci_s)
    return np.array(jpm_scores),np.array(bci_scores),inf_count



def load_result_data_v2(txt_path):
    jpm_scores = []
    bci_scores = []
    inf_count = 0
    with open(txt_path, "r") as f:
        for idx,line in enumerate(f.readlines()):
            if idx == 0 or idx == 1:
                continue
            contents = line.split('\t')[1].split(';')
            jpm_s = np.array([float(s) for s in contents[0].split(':')[1].split(',')])
            bci_s = np.array([float(s) for s in contents[1].split(':')[1].split(',')])
            # print(jpm_s,bci_s)

            if np.isinf(jpm_s).any() or np.isinf(bci_s).any():
                inf_count += 1
                continue

            jpm_scores.append(jpm_s)
            bci_scores.append(bci_s)
    return np.array(jpm_scores),np.array(bci_scores),inf_count

#5_bci_sample_100_20210615165538
#5_jpm_sample_100_20210615165600
#6_bci_sample_100_20210615165700
#6_jpm_sample_100_20210615165639
#7_bci_sample_100_20210615165715
#7_jpm_sample_100_20210615165732
#12_bci_sample_100_20210615123555
#12_jpm_sample_100_20210615123521

#0_bci_sample_100_20210617003710
#0_jpm_sample_100_20210617003730

def plot_diff_for_one_tester(bci_sample_fitted_result_path='../results/fitted_sampling/0_bci_sample_100_20210617003710.txt',
                             jpm_sample_fitted_result_path='../results/fitted_sampling/0_jpm_sample_100_20210617003730.txt'):
    # bci_sample_fitted_result_path = '../results/fitted_sampling/0_bci_sample_100_20210617003710.txt'
    # jpm_sample_fitted_result_path = '../results/fitted_sampling/0_jpm_sample_100_20210617003730.txt'

    tester_index = int(bci_sample_fitted_result_path.split('/')[-1].split('_')[0])
    print('test_id:{}'.format(tester_index))

    jpm_scores_bcisample,bci_scores_bcisample,inf_count_bci = load_result_data_v2(bci_sample_fitted_result_path)
    jpm_scores_jpmsample,bci_scores_jpmsample,inf_count_jpm = load_result_data_v2(jpm_sample_fitted_result_path)

    print(len(jpm_scores_bcisample),len(bci_scores_bcisample),inf_count_bci)
    print(len(jpm_scores_jpmsample),len(bci_scores_jpmsample),inf_count_jpm)
    # fig,axes = plt.subplots(2, 1, figsize=(8, 8))
    # bins = 20
    # axes[0].hist(jpm_scores_bcisample,bins=bins,color='mediumorchid',alpha=0.5,label='jpm score')
    # axes[0].hist(bci_scores_bcisample,bins=bins,color='limegreen',alpha=0.5,label='bci score')
    # axes[0].legend()
    # axes[0].set_title('Fitted BCI sample with both models (number of experiment:{}, include {} inf result)'.format(len(jpm_scores_bcisample)+inf_count_bci,inf_count_bci))
    # axes[1].hist(jpm_scores_jpmsample,bins=bins,color='mediumorchid',alpha=0.5,label='jpm score')
    # axes[1].hist(bci_scores_jpmsample,bins=bins,color='limegreen',alpha=0.5,label='bci score')
    # axes[1].legend()
    # axes[1].set_title('Fitted JPM sample with both models (number of experiment:{}, include {} inf result)'.format(len(bci_scores_jpmsample)+inf_count_jpm,inf_count_jpm))
    # fig.show()

    plt.figure(figsize=(6,6))
    index = 0
    percentage_diff_score_bci = (np.array(jpm_scores_bcisample[:,index])-np.array(bci_scores_bcisample[:,index]))/np.array(bci_scores_bcisample[:,index])
    percentage_diff_score_jpm = (np.array(jpm_scores_jpmsample[:,index])-np.array(bci_scores_jpmsample[:,index]))/np.array(bci_scores_jpmsample[:,index])

    diff_score_bci = np.array(jpm_scores_bcisample[:,index])-np.array(bci_scores_bcisample[:,index])
    diff_score_jpm = np.array(jpm_scores_jpmsample[:,index])-np.array(bci_scores_jpmsample[:,index])


    plt.hist(percentage_diff_score_bci,bins=20,color=COLOR_BCI,alpha=0.5,label='BCI sample' )
    plt.hist(percentage_diff_score_jpm,bins=20,color=COLOR_JPM,alpha=0.5,label='JPM sample' )
    plt.vlines(0.0,0,20,color='grey',linestyles='dashed',alpha=0.9)
    plt.xlim(-0.5,0.5)
    plt.ylim(0,20)
    plt.title('The difference between JPM score and BCI score for both samples\n($diff = \\frac{s_{JPM}- s_{BCI}}{s_{BCI}}$, score ($s$) is neg-log-likelihood)')
    plt.text(x=0,y=-2,s='PARAMs: {tester_number:12, sample_size_unit:25, N_experiment:100,\nV_snr:high, A_snr:low, snr:asynch}',
             ha='center',va='center')
    plt.legend()
    plt.savefig('../results/plots/diff_index{}_{}.png'.format(index,tester_index))
    plt.show()

    # confusion matrix
    print('\t\tFitted JPM is better\tFitted BCI is better')
    jpm_better = np.sum(jpm_scores_jpmsample[:,index]<bci_scores_jpmsample[:,index])
    bci_better = len(jpm_scores_jpmsample) - jpm_better
    print("JPM sample\t{}\t{}".format(jpm_better,bci_better))
    jpm_better = np.sum(jpm_scores_bcisample[:,index]<bci_scores_bcisample[:,index])
    bci_better = len(jpm_scores_bcisample) - jpm_better
    print("BCI sample\t{}\t{}".format(jpm_better,bci_better))




def plot_diff_for_all_tester():
    num_of_tester = 16
    N_experiment = 100

    # s_index: type of score
    # 0: only fusion neg-log-likelihood; 1: sum of all conditions' neg-log-likelihood; 2: 1 + regularization term
    s_index = 0
    s_index_explain = ['$s = -\log{(L_{fusion})}$','$s = \sum_{c\in{C}}-\log{(L_c)}$','$s = \sum_{c\in{C}}-\log{L_c} + regularization\_term$']

    # get all testers' data
    fitted_result_dir = '../results/fitted_sampling'
    f_list = os.listdir(fitted_result_dir)

    data_jpm_sample = np.zeros((num_of_tester, N_experiment))
    data_bci_sample = np.zeros((num_of_tester, N_experiment))

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
            print(inf_count)
            data_jpm_sample[test_id,:N_experiment-inf_count] = (jpm_scores[:,s_index]-bci_scores[:,s_index])/bci_scores[:,s_index]
        elif sample_type == 'bci':
            jpm_scores, bci_scores, inf_count = load_result_data_v2(fpath)
            # data_bci_sample[test_id, :] = jpm_scores[:, s_index] - bci_scores[:, s_index]
            data_bci_sample[test_id,:] = (jpm_scores[:,s_index]-bci_scores[:,s_index])/bci_scores[:,s_index]

    subplot_row = 4
    subplot_col = 4
    xlim = [0.25,0.25,0.5,0.25,0.5,0.25,0.5,0.25,1e-6,0.5,0.5,0.5,0.25,1e-6,0.5,1e-6]
    xlim = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1e-5, 0.5, 0.5, 0.5, 0.5, 1e-5, 0.5, 1e-5]
    fig, axes = plt.subplots(subplot_row, subplot_col, figsize=(12, 12))


    for i in range(num_of_tester):
        # get axes index
        axes_idx_row = int(math.floor(i / subplot_col))
        axes_idx_col = i % subplot_col
        test_id = i

        tmp_data_bci = data_bci_sample[test_id,:]
        tmp_data_jpm = data_jpm_sample[test_id,:]

        axes[axes_idx_row, axes_idx_col].hist(data_bci_sample[test_id,:], bins=np.arange(-xlim[i],xlim[i],xlim[i]/50), color=COLOR_BCI, alpha=0.5, label='BCI sample')
        axes[axes_idx_row, axes_idx_col].hist(data_jpm_sample[test_id,:], bins=np.arange(-xlim[i],xlim[i],xlim[i]/50), color=COLOR_JPM, alpha=0.5, label='JPM sample')

        axes[axes_idx_row, axes_idx_col].vlines(0.0, 0, 20, color='grey', linestyles='dashed', alpha=0.9)
        axes[axes_idx_row, axes_idx_col].set_xlim(-xlim[i],xlim[i])
        axes[axes_idx_row, axes_idx_col].set_ylim(0, 20)
        axes[axes_idx_row, axes_idx_col].set_title('tester id:{}'.format(test_id))
        axes[axes_idx_row, axes_idx_col].legend()

    fig.suptitle('The difference between JPM score and BCI score for both samples\n($diff = \\frac{s_{JPM}- s_{BCI}}{s_{BCI}}$,'+str(s_index_explain[s_index])+')',fontsize=18)
    fig.text(x=0, y=-2,
             s='PARAMs: [sample_size_unit:25, N_experiment:100, s_index:{}\nV_snr:high, A_snr:low, snr:asynch]'.format(s_index),
             ha='center', va='center')
    fig.show()
    return None



if __name__ == '__main__':
    # plot one tester
    # plot_diff_for_one_tester()

    # plot all tester
    plot_diff_for_all_tester()