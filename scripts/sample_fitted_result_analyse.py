import matplotlib.pyplot as plt
import numpy as np

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



bci_sample_fitted_result_path = '../results/bci_sample_100_20210611005532.txt'
jpm_sample_fitted_result_path = '../results/jpm_sample_100_20210611123741.txt'


jpm_scores_bcisample,bci_scores_bcisample,inf_count_bci = load_result_data(bci_sample_fitted_result_path)
jpm_scores_jpmsample,bci_scores_jpmsample,inf_count_jpm = load_result_data(jpm_sample_fitted_result_path)

fig,axes = plt.subplots(2, 1, figsize=(8, 8))
bins = 20
axes[0].hist(jpm_scores_bcisample,bins=bins,color='mediumorchid',alpha=0.5,label='jpm score')
axes[0].hist(bci_scores_bcisample,bins=bins,color='limegreen',alpha=0.5,label='bci score')
axes[0].legend()
axes[0].set_title('Fitted BCI sample with both models (number of experiment:{}, include {} inf result)'.format(len(jpm_scores_bcisample)+inf_count_bci,inf_count_bci))
axes[1].hist(jpm_scores_jpmsample,bins=bins,color='mediumorchid',alpha=0.5,label='jpm score')
axes[1].hist(bci_scores_jpmsample,bins=bins,color='limegreen',alpha=0.5,label='bci score')
axes[1].legend()
axes[1].set_title('Fitted JPM sample with both models (number of experiment:{}, include {} inf result)'.format(len(bci_scores_jpmsample)+inf_count_jpm,inf_count_jpm))
fig.show()

plt.figure(figsize=(6,6))
plt.hist(np.array(jpm_scores_bcisample)-np.array(bci_scores_bcisample),bins=20,color='limegreen',alpha=0.5,label='BCI sample' )
plt.hist(np.array(jpm_scores_jpmsample)-np.array(bci_scores_jpmsample),bins=20,color='mediumorchid',alpha=0.5,label='JPM sample' )
plt.title('The difference between JPM score and BCI score for both samples\n($diff = s_{JPM}- s_{BCI}$, score ($s$) is neg-log-likelihood)')
plt.text(x=-3,y=-1.5,s='PARAMs: {tester_number:12, sample_size_unit:25, N_experiment:100,\nV_snr:high, A_snr:low, snr:asynch}',
         ha='center',va='center')
plt.legend()
plt.show()

# confusion matrix
print('\t\tFitted JPM is better\tFitted BCI is better')
jpm_better = np.sum(jpm_scores_jpmsample<bci_scores_jpmsample)
bci_better = len(jpm_scores_jpmsample) - jpm_better
print("JPM sample\t{}\t{}".format(jpm_better,bci_better))
jpm_better = np.sum(jpm_scores_bcisample<bci_scores_bcisample)
bci_better = len(jpm_scores_bcisample) - jpm_better
print("BCI sample\t{}\t{}".format(jpm_better,bci_better))
