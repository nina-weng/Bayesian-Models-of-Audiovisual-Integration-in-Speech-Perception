import numpy as np
import scipy.stats
from scipy.special import factorial

def get_av(mu_a,mu_v,sigma_a,sigma_v,sigma_0=0):
    '''
    compute the mu and sigma for audiovisual perception
    '''
    w = (2*sigma_0**2+sigma_v**2)/(2*sigma_0**2+sigma_v**2 + sigma_a**2)
    mu_av = w*mu_a + (1-w)*mu_v
    sigma_av = np.sqrt((sigma_a**2)*w)
    return mu_av,sigma_av


def guassian2prob(mu, sigma, c):
    """
    input:
    - mu: mean of truncated Gaussian,shape [m * 1]
    - sigma: shape [m * 1]
    - c: the intervals [m * 2] or uniformed ([1 * 2])
    it could be same for all stimuli, or could be different for different stimuli

    output:
    - response_probs: [m * 3] responsed probabilities
    """
    m = len(mu)
    response_probs = np.zeros((m, 3))
    low = -5
    high = 5
    c = c.reshape(-1, 2)
    #     print(c.shape)

    if len(c) == 1:
        c = np.tile(c, (m, 1))
    #     print(c.shape)
    #     print(c)
    #     print(c[0,:])

    low_boundary = scipy.stats.norm.cdf(low, mu, sigma)
    up_boundary = scipy.stats.norm.cdf(high, mu, sigma)
    normalization_constant = up_boundary - low_boundary

    resp_bound_low = scipy.stats.norm.cdf(c[:, 0], mu, sigma)
    resp_bound_high = scipy.stats.norm.cdf(c[:, 1], mu, sigma)

    response_probs[:, 0] = np.exp(np.log(resp_bound_low - low_boundary) - np.log(normalization_constant))
    response_probs[:, 1] = np.exp(np.log(resp_bound_high - resp_bound_low) - np.log(normalization_constant))
    response_probs[:, 2] = np.exp(np.log(up_boundary - resp_bound_high) - np.log(normalization_constant))

    return response_probs


def log_max_likelihood_each(counts, mu, sigma, c):
    '''
    compute the maximum log likelihood

    input:
    - counts: shape of [m, 3]
    - mu, sigma: corresponding mu and sigma,shape [m, 1] [m, 1]
    - c: the 2 boundaries, shape of [1,2]
    '''
    x = np.array(counts)

    # get the response prob first
    res_prob = guassian2prob(mu, sigma, c)

    N = np.sum(x, axis=1)

    log_N_fact = np.log(factorial(N))

    sum_log_fact = np.sum(np.log(factorial(x)), axis=1)

    log_res_prob = np.log(res_prob)
    log_res_prob[np.isinf(log_res_prob)] = -1e6

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
    mu_Vhi_Ahi, sigma_Vhi_Ahi = get_av(mu_1, mu_2, sigma_vh, sigma_ah,sigma_0=sigma_0)
    mu_Vmid_Ahi, sigma_Vmid_Ahi = get_av(mu_1, mu_2, sigma_vm, sigma_ah,sigma_0=sigma_0)
    mu_Vlo_Ahi, sigma_Vlo_Ahi = get_av(mu_1, mu_2, sigma_vl, sigma_ah,sigma_0=sigma_0)
    mu_Vhi_Amid, sigma_Vhi_Amid = get_av(mu_1, mu_2, sigma_vh, sigma_am,sigma_0=sigma_0)
    mu_Vhi_Alo, sigma_Vhi_Alo = get_av(mu_1, mu_2, sigma_vh, sigma_al,sigma_0=sigma_0)

    mus = np.array([mu_Vhi_Ahi, mu_Vmid_Ahi, mu_Vlo_Ahi, mu_Vhi_Amid, mu_Vhi_Alo])
    sigmas = np.array([sigma_Vhi_Ahi, sigma_Vmid_Ahi, sigma_Vlo_Ahi, sigma_Vhi_Amid, sigma_Vhi_Alo])
    return mus, sigmas



def neg_log_guassian(x0, tester_index, data, model, implementation):

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
        sigma_0_s,sigma_0_a = 0, sigma_0
        [mu_vg, mu_vb, mu_ag, mu_ab] = x0[1:5]
        [sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al] = x0[5:11]
        c = x0[11:]

    else:
        raise Exception('function not implemented')


    # pre-processing the parameters



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
    # neg_log.append(-res)
    return -(res)


