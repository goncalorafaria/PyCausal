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
           # l.append(
           #     torch.nn.BatchNorm1d(n_hidden[i])
           # )
        
        l = l + [torch.nn.Linear(n_hidden[nh-2],n_hidden[nh-1]),act]
        self.z_h = torch.nn.Sequential( *l )
        
        self.z_pi = torch.nn.Linear(n_hidden[-1], n_components)
        self.z_mu = torch.nn.Linear(n_hidden[-1], n_components)
        self.z_sigma = torch.nn.Linear(n_hidden[-1], n_components)
    
    def forward(self, x, show=False):
        z_h = self.z_h(x)
        if show :
            print(z_h)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        
        sigma = torch.nn.ELU()(self.z_sigma(z_h)) + 1.00001
        #sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma
        
        #map loss    
    def maploss(pi_mu_sigma, y, reduce=True, entropy_reg=True):
        pi, mu, sigma = pi_mu_sigma
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        
        log_prob_y = m.log_prob(y) ## y | theta
        
        lp = torch.log(pi + 1e-32)
        
        log_prob_pi_y = log_prob_y + lp 
        loss = -torch.logsumexp(log_prob_pi_y, dim=1) # log ( sum_i (exp xi) )
        
        if entropy_reg:
            entropy = -torch.sum(lp * pi,dim=1)/ pi.shape[1]
            asip = torch.sum(pi * sigma, dim=-1) 
            sigp = F.softmax( sigma, dim = -1) 
            sigent = - torch.sum( sigp * torch.log(sigp) , -1 )/ sigma.shape[1]
           # torch.min(tensor1, torch.ones_like(tensor1))
            
            loss = loss - entropy - torch.square( torch.min(asip, 1*torch.ones_like(asip) )) + 1 - sigent # + sig * 4
        
        if reduce:
            return torch.mean(loss)
        else:
            return loss

    def emloss(pi_mu_sigma, y, reduce=True, entropy_reg=True):
        pi, mu, sigma = pi_mu_sigma
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob_y = m.log_prob(y) ## y | theta
        lp = torch.log(pi + 1e-32)

        log_prob_pi_y = log_prob_y + lp
        prob_pi_y = torch.exp(log_prob_pi_y)
 
        ai = prob_pi_y /( torch.sum( prob_pi_y, dim=1, keepdim=True) + 0.000001 )
        
        loss = -torch.sum( ai * log_prob_pi_y, dim = 1)

        if reduce:
            return torch.mean(loss)
        else:
            return loss

    def loss( pi_mu_sigma, y, reduce=True, entropy_reg=False, loss_type="EM"):
        if loss_type == "EM":
            return MDN.emloss(pi_mu_sigma ,y ,reduce, entropy_reg)
        elif loss_type == "MAP" :
            return MDN.maploss (pi_mu_sigma, y ,reduce, entropy_reg)
        else :
            raise Exception("Loss not implemented yet")
    
    def fit(self, scm, features="X", labels="Y", lr=1e-3, batch=248, epoch = 300, loss_type="EM"):

        optim = torch.optim.AdamW(self.parameters(), lr=lr)

        lossap = []

        for i in range(epoch):
            smps = scm._sample(batch)
            X_train = torch.tensor(smps[features], dtype=torch.float32)
            Y_train = torch.tensor(smps[labels], dtype=torch.float32)

            y_h = self.forward(X_train)
            energy = MDN.loss(y_h, Y_train, loss_type)
            optim.zero_grad()
            energy.backward()
           # torch.nn.utils.clip_grad_norm_(self.parameters(), )
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
        
#        print(pis)
#        print(mus)
#        print(sigmas)
        
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
 
