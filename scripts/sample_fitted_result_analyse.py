import matplotlib.pyplot as plt

def load_result_data(txt_path):
    jpm_scores = []
    bci_scores = []
    with open(txt_path, "r") as f:
        for idx,line in enumerate(f.readlines()):
            if idx == 0 or idx == 1:
                continue
            contents = line.split('\t')[1]
            jpm_s = float(contents.split(',')[0].split(':')[1])
            bci_s = float(contents.split(',')[1].split(':')[1][:-2])
            # print(jpm_s,bci_s)
            jpm_scores.append(jpm_s)
            bci_scores.append(bci_s)
    return jpm_scores,bci_scores



bci_sample_fitted_result_path = '../results/bci_sample_100_20210611005532.txt'
jpm_sample_fitted_result_path = '../results/jpm_sample_100_20210611082039.txt'


jpm_scores_bcisample,bci_scores_bcisample = load_result_data(bci_sample_fitted_result_path)
jpm_scores_jpmsample,bci_scores_jpmsample = load_result_data(jpm_sample_fitted_result_path)

plt.figure(figsize=(10,6))
plt.hist(jpm_scores_bcisample,bins=20,color='mediumorchid',alpha=0.5,label='jpm score')
plt.hist(bci_scores_bcisample,bins=20,color='limegreen',alpha=0.5,label='bci score')
plt.legend()
plt.title('Fitted BCI sample with both models (number of experiment:{})'.format(len(jpm_scores_bcisample)))
plt.show()

plt.figure(figsize=(10,6))
plt.hist(jpm_scores_jpmsample,bins=10,color='mediumorchid',alpha=0.5,label='jpm score')
plt.hist(bci_scores_jpmsample,bins=10,color='limegreen',alpha=0.5,label='bci score')
plt.legend()
plt.title('Fitted JPM sample with both models (number of experiment:{})'.format(len(bci_scores_jpmsample)))
plt.show()
