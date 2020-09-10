import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

class MDN(torch.nn.Module):
    def __init__(self, n_hidden, n_components, act = torch.nn.LeakyReLU() ):
        super(MDN,self).__init__()
        self.components = n_components
        
        nh = len(n_hidden)
        l = []
        for i in range(1,nh-1):
            l.append(
                torch.nn.Linear(n_hidden[i-1],n_hidden[i]) 
            )
            l.append(
                act
            )
            #l.append(
            #    nn.BatchNorm1d(n_hidden[i])
            #)
        
        l = l + [torch.nn.Linear(n_hidden[nh-2],n_hidden[nh-1]),act]
        self.z_h = torch.nn.Sequential( *l )
        
        self.z_pi = torch.nn.Linear(n_hidden[-1], n_components)
        self.z_mu = torch.nn.Linear(n_hidden[-1], n_components)
        self.z_sigma = torch.nn.Linear(n_hidden[-1], n_components)
    
    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        
        sigma = torch.nn.ELU()(self.z_sigma(z_h)) + 1
        return pi, mu, sigma
    
    def loss(pi_mu_sigma, y, reduce=True, entropy_reg=True):
        pi, mu, sigma = pi_mu_sigma
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        
        log_prob_y = m.log_prob(y) ## y | theta
        
        lp = torch.log(pi)
        
        log_prob_pi_y = log_prob_y + lp
        loss = -torch.logsumexp(log_prob_pi_y, dim=1) # log ( sum_i (exp xi) )
        
        if entropy_reg:
            entropy = -torch.sum(lp * pi,dim=1)/ pi.shape[1]
            loss = loss - 0.5 * entropy
        
        if reduce:
            return torch.mean(loss)
        else:
            return loss
    
    def fit(self, scm, features="X", labels="Y", lr=1e-3, batch=248, epoch = 300):

        optim = torch.optim.AdamW(self.parameters(), lr=lr)

        lossap = []

        for i in range(epoch):
            smps = scm._sample(batch)
            X_train = torch.tensor(smps[features], dtype=torch.float32)
            Y_train = torch.tensor(smps[labels], dtype=torch.float32)

            y_h = self.forward(X_train)
            energy = MDN.loss(y_h, Y_train)
            optim.zero_grad()
            energy.backward()
            optim.step()

            lossap.append(energy.detach().numpy())
        
        return lossap

    def sample(self, x):

        X_train = torch.tensor(x, dtype=torch.float32)
        amount = x.shape[0]
        pis, mus, sigmas = self.forward(X_train)
        
        pis = pis.detach().numpy()
        mus = mus.detach().numpy()
        sigmas = sigmas.detach().numpy()
        
        samples = np.zeros((amount, 2))
        n_mix = self.components
        to_choose_from = np.arange(n_mix)
        for j,(weights, means, std_devs) in enumerate(zip(pis, mus, sigmas)):
            index = np.random.choice(to_choose_from, p=weights)
            samples[j,1]= stats.norm.rvs(means[index], std_devs[index],size=1)
            samples[j,0]= x[j]
            if j == amount -1:
                break

        return samples
 
