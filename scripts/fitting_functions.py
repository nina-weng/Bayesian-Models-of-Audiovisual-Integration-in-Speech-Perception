import numpy as np
import scipy.stats
from scipy.special import factorial
import math
# from scipy.special import softmax

high_b = 3
low_b = -3
LAMBDA = 7


def softmax(arr):
    """
    input: arr - type: np.array or list with shape m x n or (m,) - a vector
    First, the upper part of the softmax function is complemented with a column of ones (like np.exp(0))
    Then, standard softmax is performed
    output: array - type: np.array of shape m x n+1
    """
    try:
        x,y = np.shape(arr)
    except:
        arr = np.reshape(arr, (-1,1))
        x,y = np.shape(arr)
    e_of_elements = np.hstack((np.ones((x, 1)), np.exp(arr)))
    ee_col = np.sum(e_of_elements.T, axis=0).reshape(-1,1)
    out = e_of_elements/np.tile(ee_col, ((1,y+1)))
    return out

# math functions
def sigmoid(x):
    try:
        ans = 1 / (1 + math.exp(-x))
    except OverflowError:
        ans = float('inf')

    return ans

# functions for pre=processing:
def transfer_sigmoid(value,range=3):
    '''

    :param value: the value needed transferring
    :param range: from the original scope (0,1) to (-range,range)
    :return:
    '''
    return (2*range)*sigmoid(value) - range



def get_av(mu_a,mu_v,sigma_a,sigma_v,sigma_0=0):
    '''
    compute the mu and sigma for audiovisual perception
    '''
    if np.isnan(sigma_0) or np.isinf(sigma_0):
        return mu_a,sigma_a
    w = (2*sigma_0**2+sigma_v**2)/(2*sigma_0**2+sigma_v**2 + sigma_a**2)
    mu_av = w*mu_a + (1-w)*mu_v
    sigma_av = np.sqrt((sigma_a**2)*w)
    return mu_av,sigma_av


def guassian2prob(mu, sigma, c):
    """
    input:
    - mu: mean of truncated Gaussian,shape [m * 1]
    - sigma: shape [m * 1]
    - c: the intervals [m * (k-1)] or uniformed ([1 * (k-1)])
    k is the number of response
    it could be same for all stimuli, or could be different for different stimuli

    output:
    - response_probs: [m * k] responsed probabilities
    """
    m = len(mu)
    k = len(c)
    # print(k)
    response_probs = np.zeros((m, k+1))
    low = low_b
    high = high_b



    c = c.reshape(-1, k)
    #     print(c.shape)




    if len(c) == 1:
        c = np.tile(c, (m, 1))
    #     print(c.shape)
    #     print(c)
    #     print(c[0,:])

    low_boundary = scipy.stats.norm.cdf(low, mu, sigma)
    up_boundary = scipy.stats.norm.cdf(high, mu, sigma)
    normalization_constant = up_boundary - low_boundary

    # resp_bound_low = scipy.stats.norm.cdf(c[:, 0], mu, sigma)
    # resp_bound_high = scipy.stats.norm.cdf(c[:, 1], mu, sigma)

    resp_bounds = np.zeros((k,m))
    for i in range(k):
        resp_bounds[i,:] = scipy.stats.norm.cdf(c[:, i], mu, sigma)


    response_probs = np.zeros((m,k+1))
    for i in range(k+1):
        if i == 0:
            response_probs[:,i] =np.exp(np.log(resp_bounds[i] - low_boundary) - np.log(normalization_constant))
        elif i == k:
            response_probs[:, i] = np.exp(np.log(up_boundary - resp_bounds[i-1]) - np.log(normalization_constant))
        else:
            response_probs[:,i] = np.exp(np.log(resp_bounds[i] - resp_bounds[i-1]) - np.log(normalization_constant))

    # response_probs_2 = np.zeros((m, k + 1))
    # response_probs_2[:, 0] = np.exp(np.log(resp_bound_low - low_boundary) - np.log(normalization_constant))
    # response_probs_2[:, 1] = np.exp(np.log(resp_bound_high - resp_bound_low) - np.log(normalization_constant))
    # response_probs_2[:, 2] = np.exp(np.log(up_boundary - resp_bound_high) - np.log(normalization_constant))

    # if (resp_bound_high - resp_bound_low).any() <=0:
    #     # print(c)
    # if (up_boundary - resp_bound_high).any() <=0:
    #     print(c)

    return response_probs


def log_max_likelihood_each(counts, mu, sigma, c):
    '''
    compute the maximum log likelihood for multinomial distribution

    input:
    - counts: shape of [m, 3]
    - mu, sigma: corresponding mu and sigma,shape [m, 1] [m, 1]
    - c: the 2 boundaries, shape of [1,2]
    '''
    x = np.array(counts)
    # print(c)

    if c[-1] >= high_b or c[0] <= low_b :
        # print(c)
        return -1e6

    # get the response prob first
    res_prob = guassian2prob(mu, sigma, c)


    N = np.sum(x, axis=1)

    # log_N_fact = np.log(factorial(N))
    N_list = np.arange(1,N+1)
    log_N_fact = np.sum(np.log(N_list))

    sum_log_fact = np.sum(np.log(factorial(x)), axis=1)

    log_res_prob = np.log(res_prob)
    # log_res_prob[np.isinf(log_res_prob)] = -1e6

    # if the prob goes to 0 (log_prob -> inf/nan), then
    if np.sum(np.isinf(log_res_prob)) > 0:
        return -1e6

    sum_xi_log = np.sum(np.multiply(x, log_res_prob), axis=1)


    res = log_N_fact - sum_log_fact + sum_xi_log

    return np.sum(res)


def log_max_likelihood_bci(counts, mu_pc1, sigma_pc1, mu_pc2, sigma_pc2,c, p):
    '''
    compute the maximum log likelihood

    input:
    - counts: shape of [m, 3]
    - mu, sigma: corresponding mu and sigma,shape [m, 1] [m, 1]
    - c: the 2 boundaries, shape of [1,2]
    '''
    x = np.array(counts)
    # print(c)

    if c[-1] >= high_b or c[0] <= low_b :
        # print(c)
        return -1e6

    # get the response prob first
    res_prob_pc1 = guassian2prob(mu_pc1, sigma_pc1, c)
    res_prob_pc2 = guassian2prob(mu_pc2, sigma_pc2, c)

    res_prob = p*res_prob_pc1 + (1-p)*res_prob_pc2


    N = np.sum(x, axis=1)

    # log_N_fact = np.log(factorial(N))
    N_list = np.arange(1, N + 1)
    log_N_fact = np.sum(np.log(N_list))

    sum_log_fact = np.sum(np.log(factorial(x)), axis=1)

    log_res_prob = np.log(res_prob)
    # log_res_prob[np.isinf(log_res_prob)] = -1e6

    # if the prob goes to 0 (log_prob -> inf/nan), then
    if np.sum(np.isinf(log_res_prob)) > 0:
        return -1e6

    sum_xi_log = np.sum(np.multiply(x, log_res_prob), axis=1)


    res = log_N_fact - sum_log_fact + sum_xi_log

    return np.sum(res)


def get_params_AV(mu_1, mu_2, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am,
                  sigma_al,sigma_0):  # ,sigma_vh,sigma_vm,sigma_vl,sigma_ah,sigma_am,sigma_al
    '''
    compute the array of mu and sigma for multi-sesories
    input:
    - mu_1 : mu for audio
    - mu_2 : mu for visual
    return:
    the array of mu and sigma of the defined-combinations in experiment
    '''
    # get the mu_av, sigma_av

    # Vhi_Ahi,Vmid_Ahi,Vlo_Ahi,Vhi_Amid,Vhi_Alo
    mu_Vhi_Ahi, sigma_Vhi_Ahi = get_av(mu_1, mu_2, sigma_ah,sigma_vh,sigma_0=sigma_0)
    mu_Vmid_Ahi, sigma_Vmid_Ahi = get_av(mu_1, mu_2,sigma_ah,sigma_vm,sigma_0=sigma_0)
    mu_Vlo_Ahi, sigma_Vlo_Ahi = get_av(mu_1, mu_2, sigma_ah, sigma_vl,sigma_0=sigma_0)
    mu_Vhi_Amid, sigma_Vhi_Amid = get_av(mu_1, mu_2, sigma_am, sigma_vh,sigma_0=sigma_0)
    mu_Vhi_Alo, sigma_Vhi_Alo = get_av(mu_1, mu_2, sigma_al, sigma_vh,sigma_0=sigma_0)

    mus = np.array([mu_Vhi_Ahi, mu_Vmid_Ahi, mu_Vlo_Ahi, mu_Vhi_Amid, mu_Vhi_Alo])
    sigmas = np.array([sigma_Vhi_Ahi, sigma_Vmid_Ahi, sigma_Vlo_Ahi, sigma_Vhi_Amid, sigma_Vhi_Alo])
    return mus, sigmas



def neg_log_guassian(x0, tester_index, data, model, implementation, preprocess = False, coef_lambda = LAMBDA):

    # not implemented for other model
    # JPM full implementation
    sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw = x0[6:12]

    preprocess_x0 = parameter_prepocess(x0,model,implementation,preprocess)

    sigma_0_s, sigma_0_a = preprocess_x0[0:2]
    [mu_vg, mu_vb, mu_ag, mu_ab] = preprocess_x0[2:6]
    [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = preprocess_x0[6:12]
    c = preprocess_x0[12:]



    # data['AB']
    res_ab = log_max_likelihood_each(data['AB']['counts'][tester_index, :, :],
                                     np.array([mu_ab] * 3),
                                     np.array([sigma_ah, sigma_am, sigma_al]), c)

    # data['AG']
    res_ag = log_max_likelihood_each(data['AG']['counts'][tester_index, :, :],
                                     np.array([mu_ag] * 3),
                                     np.array([sigma_ah, sigma_am, sigma_al]), c)

    # data['VB']
    res_vb = log_max_likelihood_each(data['VB']['counts'][tester_index, :, :],
                                     np.array([mu_vb] * 3),
                                     np.array([sigma_vh, sigma_vm, sigma_vl]), c)

    # data['VG']
    res_vg = log_max_likelihood_each(data['VG']['counts'][tester_index, :, :],
                                     np.array([mu_vg] * 3),
                                     np.array([sigma_vh, sigma_vm, sigma_vl]), c)

    # SYNCH part
    # data['AVFus']['synch']
    # AVFus -> auditory B , visual G
    mus, sigmas = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,sigma_0=sigma_0_s)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_s = log_max_likelihood_each(data['AVFus']['synch']['counts'][tester_index, :, :],
                                        mus, sigmas, c)

    # data['AVG']['synch']
    mus, sigmas = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,sigma_0=sigma_0_s)
    res_avg_s = log_max_likelihood_each(data['AVG']['synch']['counts'][tester_index, :, :],
                                      mus, sigmas, c)

    # data['AVB']['synch']
    mus, sigmas = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,sigma_0=sigma_0_s)
    res_avb_s = log_max_likelihood_each(data['AVB']['synch']['counts'][tester_index, :, :],
                                      mus, sigmas, c)

    # ASYNCH part
    # data['AVFus']['asynch']
    # AVFus -> auditory B , visual G
    mus, sigmas = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_a = log_max_likelihood_each(data['AVFus']['asynch']['counts'][tester_index, :, :],
                                          mus, sigmas, c)

    # data['AVG']['asynch']
    mus, sigmas = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    res_avg_a = log_max_likelihood_each(data['AVG']['asynch']['counts'][tester_index, :, :],
                                        mus, sigmas, c)

    # data['AVB']['asynch']
    mus, sigmas = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    res_avb_a = log_max_likelihood_each(data['AVB']['asynch']['counts'][tester_index, :, :],
                                        mus, sigmas, c)


    res = res_ab + res_ag + res_vb + res_vg + res_avfus_s + res_avg_s + res_avb_s +\
          res_avfus_a + res_avg_a + res_avb_a

    # set the regularization term as the penalty on big sigmas
    # Regularization method 1
    sigs_raw = np.array([sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw])
    regularization_term = coef_lambda * np.sum(sigs_raw ** 2)

    # Regularization method 2
    sigs = np.array([sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
    regularization_term = coef_lambda * np.sum((1 / sigs) ** 2)

    whole_res = -res + regularization_term

    # if c[0] <-3.5 or c[1] > 3.5:
    #     #     print(c)

    # print(res_ab,res_ag,res_vb,res_vg,res_avfus_s,res_avg_s,res_avb_s,res_avfus_a,res_avg_a,res_avb_a)
    # neg_log.append(-res)

    # for debug
    # print('res_ab:{}\tres_ag:{}\tres_vb:{}\tres_vg:{}'.format(res_ab,res_ag,res_vb,res_vg))
    # print('res_avb_s:{}\tres_avb_a:{}'.format(res_avb_s,res_avb_a))
    # print('res_avg_s:{}\tres_avg_a:{}'.format(res_avg_s, res_avg_a))
    # print('res_avfus_s:{}\tres_avfus_a:{}'.format(res_avfus_s, res_avfus_a))
    # print('regularization_term:{}'.format(regularization_term))

    return whole_res

def neg_log_bci(x0, tester_index, data, model, implementation, preprocess = False, coef_lambda = LAMBDA):
    sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw = x0[6:12]
    # x0 = p_s, p_a, mu_vg, mu_vb, mu_ag, mu_ab,sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,c
    preprocess_x0 = parameter_prepocess(x0, model, implementation, preprocess)

    p_s,p_a = preprocess_x0[0:2]
    [mu_vg, mu_vb, mu_ag, mu_ab] = preprocess_x0[2:6]
    [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = preprocess_x0[6:12]
    c = preprocess_x0[12:]

    # data['AB']
    res_ab = log_max_likelihood_each(data['AB']['counts'][tester_index, :, :],
                                     np.array([mu_ab] * 3),
                                     np.array([sigma_ah, sigma_am, sigma_al]), c)

    # data['AG']
    res_ag = log_max_likelihood_each(data['AG']['counts'][tester_index, :, :],
                                     np.array([mu_ag] * 3),
                                     np.array([sigma_ah, sigma_am, sigma_al]), c)

    # data['VB']
    res_vb = log_max_likelihood_each(data['VB']['counts'][tester_index, :, :],
                                     np.array([mu_vb] * 3),
                                     np.array([sigma_vh, sigma_vm, sigma_vl]), c)

    # data['VG']
    res_vg = log_max_likelihood_each(data['VG']['counts'][tester_index, :, :],
                                     np.array([mu_vg] * 3),
                                     np.array([sigma_vh, sigma_vm, sigma_vl]), c)

    # SYNCH part
    # data['AVFus']['synch']
    # AVFus -> auditory B , visual G
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_s = log_max_likelihood_bci(data['AVFus']['synch']['counts'][tester_index, :, :],
                                         mus_pc1, sigmas_pc1,mus_pc2, sigmas_pc2, c, p_s)

    # data['AVG']['synch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avg_s = log_max_likelihood_bci(data['AVG']['synch']['counts'][tester_index, :, :],
                                        mus_pc1, sigmas_pc1, mus_pc2, sigmas_pc2, c, p_s)

    # data['AVB']['synch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avb_s = log_max_likelihood_bci(data['AVB']['synch']['counts'][tester_index, :, :],
                                        mus_pc1, sigmas_pc1, mus_pc2, sigmas_pc2, c, p_s)

    # ASYNCH part
    # data['AVFus']['asynch']
    # AVFus -> auditory B , visual G
    mus_pc1, sigmas_pc1= get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_a = log_max_likelihood_bci(data['AVFus']['asynch']['counts'][tester_index, :, :],
                                          mus_pc1, sigmas_pc1, mus_pc2, sigmas_pc2, c, p_a)

    # data['AVG']['asynch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avg_a = log_max_likelihood_bci(data['AVG']['asynch']['counts'][tester_index, :, :],
                                        mus_pc1, sigmas_pc1, mus_pc2, sigmas_pc2, c, p_a)

    # data['AVB']['asynch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avb_a = log_max_likelihood_bci(data['AVB']['asynch']['counts'][tester_index, :, :],
                                       mus_pc1, sigmas_pc1, mus_pc2, sigmas_pc2, c, p_a)

    # res = res_ab + res_ag + res_vb + res_vg + 2 * res_avfus_s + 2 * res_avg_s + 2 * res_avb_s + \
    #       2 * res_avfus_a + 2 * res_avg_a + 2 * res_avb_a
    res = res_ab + res_ag + res_vb + res_vg + res_avfus_s +  res_avg_s + res_avb_s + \
          res_avfus_a + res_avg_a + res_avb_a

    # set the regularization term as the penalty on big sigmas
    # Regularization method 1
    sigs_raw = np.array([sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw])
    regularization_term = coef_lambda * np.sum(sigs_raw ** 2)

    # Regularization method 2
    sigs = np.array([sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
    regularization_term = coef_lambda * np.sum((1/sigs) ** 2)

    whole_res = -res + regularization_term

    return whole_res



def parameter_prepocess( x0,model, implementation,preprocess=False):
    # get the parameter out from x0
    if model == 'JPM' and implementation == 'full':
        if len(x0) != 14:
            raise Exception('length of parameters do not match model.')
        sigma_0_s, sigma_0_a = x0[0:2]
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
        c = x0[12:]

    elif model == 'JPM' and implementation == 'reduced':
        if len(x0) != 13:
            raise Exception('length of parameters do not match model.')
        sigma_0 = x0[0]
        sigma_0_s, sigma_0_a = 0, sigma_0
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[1:5]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[5:11]
        c = x0[11:]

    elif model == 'JPM' and implementation == 'mle':
        if len(x0) != 12:
            raise Exception('length of parameters do not match model.')
        sigma_0_s, sigma_0_a = 0, 0
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[0:4]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[4:10]
        c = x0[10:]

    elif model == 'BCI' and implementation == 'full':
        if len(x0) != 14:
            raise Exception('length of parameters do not match model.')
        p_s, p_a = x0[0:2] # p_s,p_a
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
        c = x0[12:]

    else:
        raise Exception('function not implemented')

    # pre-processing the parameters
    if preprocess:

        if model == 'BCI':
            [p_s,p_a] = [sigmoid(p) for p in np.array([p_s,p_a])]

        # for all mus -> use sigmoid to convert into range (-3,3)
        [mu_vg, mu_vb, mu_ag, mu_ab] = [transfer_sigmoid(mu, range=3) for mu in np.array([mu_vg, mu_vb, mu_ag, mu_ab])]
        # for sigma -> use exponential function to convert sigma to (1, +inf)
        sigs = [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al]
        # method 0: [np.sqrt(np.exp(sig ** 2)) for sig in sigs]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = [np.exp(sig) for sig in
                                                                        sigs]  # np.exp(np.abs(sig))
        # intervals/boundaries
        softmaxBounds = softmax([c])
        c = 6 * (np.cumsum(softmaxBounds) - 0.5)

    if model == 'JPM':
        new_x0 = np.array([sigma_0_s, sigma_0_a,mu_vg, mu_vb, mu_ag, mu_ab,
            sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,c[0],c[1]])
    elif model == 'BCI':
        new_x0 = np.array([p_s,p_a, mu_vg, mu_vb, mu_ag, mu_ab,
                           sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al, c[0], c[1]])

    return new_x0


# mu_vg, mu_vb, mu_ag, mu_ab = -5,-1,2.5,6
# [mu_vg, mu_vb, mu_ag, mu_ab] = [transfer_sigmoid(mu) for mu in np.array([mu_vg, mu_vb, mu_ag, mu_ab])]#transfer_sigmoid(np.array([mu_vg, mu_vb, mu_ag, mu_ab]))
# print(mu_vg, mu_vb, mu_ag, mu_ab)

# sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al = 0.01, 0.2,0.75,4,2,0.04
# sigs = [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al]
# [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = [np.exp(np.abs(sig))  for sig in sigs]
# print(sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al )
#
# c = [0,0]
# softmaxBounds = softmax(c)
# print(softmaxBounds)
# print(np.cumsum(softmaxBounds))
# c = 6*(np.cumsum(softmaxBounds)-0.5)
# print(c)

#
# sigs = np.array([sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
# regularization_term = np.sum(sigs**2)
# print(regularization_term)

# print(get_av(1,0.5,1.2,3,np.inf))


def neg_log_jpm_for1tester(x0, data_sample, model, implementation, preprocess = False, coef_lambda = LAMBDA):


    sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw = x0[6:12]

    preprocess_x0 = parameter_prepocess_9cates(x0,model,implementation,preprocess)

    sigma_0_s, sigma_0_a = preprocess_x0[0:2]
    [mu_vg, mu_vb, mu_ag, mu_ab] = preprocess_x0[2:6]
    [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = preprocess_x0[6:12]
    c = preprocess_x0[12:]

    # SNR: V-high, A-low
    sigma_a = sigma_al
    sigma_v = sigma_vh
    snr_type = 4



    # data['AB']
    res_ab = log_max_likelihood_each(data_sample['AB']['counts'].reshape(1,-1),
                                     np.array([mu_ab]),
                                     np.array([sigma_a]), c)

    # data['AG']
    res_ag = log_max_likelihood_each(data_sample['AG']['counts'].reshape(1,-1),
                                     np.array([mu_ag]),
                                     np.array([sigma_a]), c)

    # data['VB']
    res_vb = log_max_likelihood_each(data_sample['VB']['counts'].reshape(1,-1),
                                     np.array([mu_vb]),
                                     np.array([sigma_v]), c)

    # data['VG']
    res_vg = log_max_likelihood_each(data_sample['VG']['counts'].reshape(1,-1),
                                     np.array([mu_vg]),
                                     np.array([sigma_v]), c)


    # ASYNCH part
    # data['AVFus']['asynch']
    # AVFus -> auditory B , visual G
    mus, sigmas = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_a = log_max_likelihood_each(data_sample['AVFus']['counts'].reshape(1,-1),
                                          [mus[snr_type]], [sigmas[snr_type]], c)

    # data['AVG']['asynch']
    mus, sigmas = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    res_avg_a = log_max_likelihood_each(data_sample['AVG']['counts'].reshape(1,-1),
                                        [mus[snr_type]], [sigmas[snr_type]], c)

    # data['AVB']['asynch']
    mus, sigmas = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                sigma_0=sigma_0_a)
    res_avb_a = log_max_likelihood_each(data_sample['AVB']['counts'].reshape(1,-1),
                                        [mus[snr_type]], [sigmas[snr_type]], c)


    res = res_ab + res_ag + res_vb + res_vg + \
          5*res_avfus_a + res_avg_a + res_avb_a

    # set the regularization term as the penalty on big sigmas
    # Regularization method 1
    sigs_raw = np.array([sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw])
    regularization_term = coef_lambda * np.sum(sigs_raw ** 2)

    # Regularization method 2
    sigs = np.array([sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
    regularization_term = coef_lambda * np.sum((1 / sigs) ** 2)

    whole_res = -res + regularization_term

    return whole_res


def neg_log_bci_for1tester(x0, data_sample, model, implementation, preprocess = False, coef_lambda = LAMBDA):

    sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw = x0[6:12]
    preprocess_x0 = parameter_prepocess_9cates(x0, model, implementation, preprocess)

    p_s, p_a = preprocess_x0[0:2]
    [mu_vg, mu_vb, mu_ag, mu_ab] = preprocess_x0[2:6]
    [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = preprocess_x0[6:12]
    c = preprocess_x0[12:]

    # SNR: V-high, A-low
    sigma_a = sigma_al
    sigma_v = sigma_vh
    snr_type = 4

    # data['AB']
    res_ab = log_max_likelihood_each(data_sample['AB']['counts'].reshape(1, -1),
                                     np.array([mu_ab]),
                                     np.array([sigma_a]), c)

    # data['AG']
    res_ag = log_max_likelihood_each(data_sample['AG']['counts'].reshape(1, -1),
                                     np.array([mu_ag]),
                                     np.array([sigma_a]), c)

    # data['VB']
    res_vb = log_max_likelihood_each(data_sample['VB']['counts'].reshape(1, -1),
                                     np.array([mu_vb]),
                                     np.array([sigma_v]), c)

    # data['VG']
    res_vg = log_max_likelihood_each(data_sample['VG']['counts'].reshape(1, -1),
                                     np.array([mu_vg]),
                                     np.array([sigma_v]), c)


    # ASYNCH part
    # data['AVFus']['asynch']
    # AVFus -> auditory B , visual G
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    #     print('AVFus,mus:{}'.format(mus))
    #     print('AVFus,sigmas:{}'.format(sigmas))
    res_avfus_a = log_max_likelihood_bci(data_sample['AVG']['counts'].reshape(1,-1),
                                         [mus_pc1[snr_type]], [sigmas_pc1[snr_type]],
                                         [mus_pc2[snr_type]], [sigmas_pc2[snr_type]], c, p_a)

    # data['AVG']['asynch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ag, mu_vg, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avg_a = log_max_likelihood_bci(data_sample['AVG']['counts'].reshape(1,-1),
                                       [mus_pc1[snr_type]], [sigmas_pc1[snr_type]],
                                       [mus_pc2[snr_type]], [sigmas_pc2[snr_type]], c, p_a)

    # data['AVB']['asynch']
    mus_pc1, sigmas_pc1 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=0)
    mus_pc2, sigmas_pc2 = get_params_AV(mu_ab, mu_vb, sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,
                                        sigma_0=np.inf)
    res_avb_a = log_max_likelihood_bci(data_sample['AVB']['counts'].reshape(1,-1),
                                       [mus_pc1[snr_type]], [sigmas_pc1[snr_type]],
                                       [mus_pc2[snr_type]], [sigmas_pc2[snr_type]], c, p_a)

    res = res_ab + res_ag + res_vb + res_vg + \
          5*res_avfus_a + res_avg_a + res_avb_a

    # set the regularization term as the penalty on big sigmas
    # Regularization method 1
    sigs_raw = np.array([sigma_vh_raw, sigma_vm_raw, sigma_vl_raw, sigma_ah_raw, sigma_am_raw, sigma_al_raw])
    regularization_term = coef_lambda * np.sum(sigs_raw ** 2)

    # Regularization method 2
    sigs = np.array([sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
    regularization_term = coef_lambda * np.sum((1 / sigs) ** 2)

    whole_res = -res + regularization_term

    return whole_res


def parameter_prepocess_9cates(x0,model, implementation,preprocess=False):
    # get the parameter out from x0
    if model == 'JPM' and implementation == 'full':
        if len(x0) != 20:
            raise Exception('length of parameters do not match model.')
        sigma_0_s, sigma_0_a = x0[0:2]
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
        c = x0[12:]

    elif model == 'BCI' and implementation == 'full':
        if len(x0) != 20:
            raise Exception('length of parameters do not match model.')
        p_s, p_a = x0[0:2] # p_s,p_a
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[2:6]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[6:12]
        c = x0[12:]

    else:
        raise Exception('function not implemented')

    # pre-processing the parameters
    if preprocess:

        if model == 'BCI':
            [p_s,p_a] = [sigmoid(p) for p in np.array([p_s,p_a])]

        # for all mus -> use sigmoid to convert into range (-3,3)
        [mu_vg, mu_vb, mu_ag, mu_ab] = [transfer_sigmoid(mu, range=3) for mu in np.array([mu_vg, mu_vb, mu_ag, mu_ab])]
        # for sigma -> use exponential function to convert sigma to (1, +inf)
        sigs = [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al]
        # method 0: [np.sqrt(np.exp(sig ** 2)) for sig in sigs]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = [np.exp(sig) for sig in
                                                                        sigs]  # np.exp(np.abs(sig))
        # intervals/boundaries
        softmaxBounds = softmax([c])
        c = 6 * (np.cumsum(softmaxBounds) - 0.5)

    if model == 'JPM':
        new_x0 = np.array([sigma_0_s, sigma_0_a,mu_vg, mu_vb, mu_ag, mu_ab,
            sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
        new_x0 = np.append(new_x0,c[:-1])
    elif model == 'BCI':
        new_x0 = np.array([p_s,p_a, mu_vg, mu_vb, mu_ag, mu_ab,
                           sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al])
        new_x0 = np.append(new_x0, c[:-1])

    return new_x0