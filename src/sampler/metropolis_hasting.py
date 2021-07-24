
import torch
import logging
import numpy as np

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# metropolis hasting algorithm: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

class MetropolisHasting(torch.nn.Module):
    def __init__(self, wave_fn, batch_size, state_size, burn_in, num_chains, init_prev_state):
        super(MetropolisHasting, self).__init__()
        self.num_chains = num_chains
        self.batch_size = batch_size
        self.wave_fn = wave_fn
        self.state_size = state_size
        self.output_stats = False
        # random imitialize the initialization
        self.init = np.random.randint(2, size=(self.num_chains, self.state_size))*2 - 1
        # determine if use the states sampled previously as initialization
        # this feature is in general good, but sometimes we disable it for meta learning
        # for meta learning, the number of iteration in the inner loop is small
        # instead a large burn_in number is used there.
        self.init_prev_state = init_prev_state
        self.burn_in = burn_in
        self.iters = int(np.ceil(self.batch_size / self.num_chains) + self.burn_in)

    @torch.no_grad()
    def forward(self):
        self.wave_fn.eval()
        states = np.zeros((self.num_chains, self.iters, self.state_size))
        if self.init_prev_state:
            curr = self.init
        else:
            curr = np.random.randint(2, size=(self.num_chains, self.state_size))*2 - 1
        avg_accept = 0
        for i in range(self.iters):
            prop = self.proposal_dist(curr)
            # for _ in range(1):
            #     prop = self.proposal_dist(prop)
            # log of unnormed probability = log(|psi|^2) = 2*log(|psi|)
            log_p_curr = self.wave_fn(torch.tensor(curr).float()).data.numpy()
            log_p_prop = self.wave_fn(torch.tensor(prop).float()).data.numpy()
            # acceptance probability
            # transition probabilities are cancelled out
            transition = log_p_prop - log_p_curr
            if transition.dtype == 'complex64':
                transition = transition.real
            probs = np.minimum(np.exp(2 * transition), 1)
            avg_accept += probs.mean() / self.iters
            accepts = self.random_coin(probs)
            nxt = [prop[j] if accepts[j] else curr[j] for j in range(len(accepts))]
            curr = np.stack(nxt)
            states[:, i, :] = curr
        states_keep = states[:, self.burn_in:, :]
        states_keep = np.reshape(states_keep, (-1, self.state_size))
        # update the initialization
        self.init = states[:, -1, :]
        if self.output_stats:
            return states_keep[:self.batch_size], avg_accept
        else:
            return states_keep[:self.batch_size]

    def proposal_dist(self, curr):
        # nearest neighbor swap with uniform probability (random walker MCMC)
        indices = np.floor(np.random.uniform(0, curr.shape[-1], self.num_chains)).astype(np.int)
        nxt = curr.copy()
        nxt[np.arange(len(nxt)), indices] *= -1
        return nxt

    def random_coin(self, p):
        unif = np.random.uniform(0, 1, self.num_chains)
        accepts = unif < p
        return accepts

    @torch.no_grad()
    def debug(self):
        self.wave_fn.eval()
        states = np.zeros((self.num_chains, self.iters, self.state_size))
        curr = self.init
        avg_accept = 0
        for i in range(self.iters):
            prop = self.proposal_dist(curr)
            # log of unnormed probability = log(|psi|^2) = 2*log(|psi|)
            log_p_curr = self.wave_fn(torch.tensor(curr).float()).data.numpy()*2
            log_p_prop = self.wave_fn(torch.tensor(prop).float()).data.numpy()*2
            # acceptance probability
            # transition probabilities are cancelled out
            transition = (log_p_prop-log_p_curr)
            probs = np.minimum(np.exp(transition), 1)
            avg_accept += probs.mean() / self.iters
            accepts = self.random_coin(probs)
            nxt = [prop[j] if accepts[j] else curr[j] for j in range(len(accepts))]
            curr = np.stack(nxt)
            states[:, i, :] = curr
        self.init = curr
        return states



# class MetropolisHasting(torch.nn.Module):
#     def __init__(self, wave_fn, batch_size, state_size, burn_in, num_chains):
#         super(MetropolisHasting, self).__init__()
#         self.iters = batch_size
#         self.wave_fn = wave_fn
#         self.state_size = state_size
#         self.burn_in = max(int(self.iters / 10), burn_in)
#         self.print_stats = False

#     def forward(self):
#         self.wave_fn.eval()
#         states = []
#         curr = np.random.randint(2, size=self.state_size)*2 - 1
#         avg_accept = 0
#         for i in range(int(self.iters + self.burn_in)):
#             nxt = self.proposal_dist(curr)
#             # log of unnormed probability = log(|psi|^2) = 2*log(|psi|)
#             curr_tensor = torch.tensor(curr).float().unsqueeze(0)
#             nxt_tensor = torch.tensor(nxt).float().unsqueeze(0)
#             log_p_curr = self.wave_fn(curr_tensor).squeeze().item()*2
#             log_p_nxt = self.wave_fn(nxt_tensor).squeeze().item()*2
#             # acceptance probability
#             # transition probabilities are cancelled out
#             acceptance = min(np.exp(log_p_nxt-log_p_curr), 1)
#             avg_accept += acceptance
#             if self.random_coin(acceptance):
#                 curr = nxt
#             states.append(curr)
#         assert len(states) == (self.iters + self.burn_in)
#         if self.print_stats:
#             logging.info("Sampled {}, Average acceptance rate {}.".format(i+1, avg_accept/(i+1)))
#         return states[self.burn_in:]

#     def proposal_dist(self, curr):
#         # nearest neighbor swap with uniform probability (random walker MCMC)
#         index = np.floor(np.random.uniform(0, curr.shape[0], 1)[0]).astype(np.int)
#         nxt = curr.copy()
#         nxt[index] = -nxt[index]
#         return nxt

#     def random_coin(self, p):
#         unif = np.random.uniform(0,1)
#         if unif >= p:
#             return False
#         else:
#             return True