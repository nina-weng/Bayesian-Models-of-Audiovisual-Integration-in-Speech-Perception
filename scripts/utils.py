import numpy as np


def get_random_free_params(model = 'JPM',implementation = 'full'):
    if implementation == 'full':
        init_para = np.random.rand(14)
        # [sigma_0_synch,sigma_0_asynch,mu_vg, mu_vb, mu_ag, mu_ab,sigma_vh, sigma_vm, sigma_vl, sigma_ah, sigma_am, sigma_al,interval_gd,interval_db]
    elif implementation == 'reduced':
        init_para = np.random.rand(13)
    elif implementation == 'mle':
        init_para = np.random.rand(12)
    else:
        raise Exception('not recognize the implementation method: {}'.format(implementation))

    return init_para


