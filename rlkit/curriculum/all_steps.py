import numpy as np
import torch
from rlkit.core import logger
from rlkit.torch.sac.policies import MakeDeterministic


def all_steps(policy,
              qf1,
              qf2,
              env,
              num_curriculum_eps=1,
              curr_k=10,
              curr_beta=0.9,
              curr_ep_length=300,
              bootstrap_value=True,
              use_cuda=True):
    if use_cuda:
        device='cuda'
    else:
        device='cpu'
    logger.log('\nStarting adaptative curriculum.\n')
    p = {}
    det_policy = MakeDeterministic(policy)
    for k, v in env.curr_grid.items():
        capabilities = []
        for i, init in enumerate(v):
            logger.log('Inicialização - Curriculum {}. Iter {} -> {}'.format(k, i, init))
            for e in range(num_curriculum_eps):
                accum_c = []
                o, d, ep_ret, ep_len = env.reset(
                    curr_init=init, init_strategy='adaptative', curr_var=k), False, 0, 0
                c = 0
                while not(d or (ep_len == curr_ep_length)):
                    # Take deterministic actions at test time
                    a, _ = det_policy.get_action(o)
                    o, r, d, _ = env.step(a)
                    if bootstrap_value:
                        o = torch.Tensor(o).to(device)
                        dist = policy(o.view(1,-1))
                        new_obs_actions, _ = dist.rsample_and_logprob()
                        q_new_actions = torch.min(
                            qf1(o.view(1,-1), new_obs_actions),
                            qf2(o.view(1,-1), new_obs_actions),
                        )
                        # Estimates value
                        v = q_new_actions.mean()
                        if not use_cuda:
                            c += v.detach().numpy()
                        else:
                            c += v.detach().cpu().numpy()
                    else:
                        # Uses returns instead
                        ep_ret += r
                        c = ep_ret
                    ep_len += 1
                accum_c.append(c)
            capabilities.append(np.mean(accum_c))
        max_capability = np.max(capabilities)
        f = np.exp(-curr_k*np.abs(np.array(capabilities) /
                                  max_capability - curr_beta))
        p[k] = f/f.sum()
    return p
