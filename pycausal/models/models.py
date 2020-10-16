import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans

class GMMOutput(torch.nn.Module):

    def __init__(self, n_components):
        super(GMMOutput, self).__init__()
        self.components = n_components

    def sample(self, x):

        X_train = x
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

    def maploss(pi_mu_sigma, y, reduce=True, entropy_reg=True, alpha=2):
        pi, mu, sigma = pi_mu_sigma
        m = torch.distributions.Normal(loc=mu, scale=sigma)

        log_prob_y = m.log_prob(y) ## y | theta

        lp = torch.log(pi)

        log_prob_pi_y = log_prob_y + lp
        loss = -torch.logsumexp(log_prob_pi_y, dim=1) # log ( sum_i (exp xi) )

        if entropy_reg:
            entropy = -torch.sum(lp * pi,dim=1)/ pi.shape[1]

            loss = loss - entropy * alpha

        if reduce:
            return torch.mean(loss)
        else:
            return loss

    def emloss(pi_mu_sigma, y, reduce=True, entropy_reg=True, alpha=2):
        pi, mu, sigma = pi_mu_sigma
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob_y = m.log_prob(y) ## y | theta
        lp = torch.log(pi)

        log_prob_pi_y = log_prob_y + lp
        #prob_pi_y = torch.exp(log_prob_pi_y)

        ai = F.softmax(log_prob_pi_y, dim=1)
        #ai = prob_pi_y /( torch.sum( prob_pi_y, dim=1, keepdim=True) + 0.000001 )

        loss = -torch.sum( ai * log_prob_pi_y, dim = 1)

        if entropy_reg:
            entropy = -torch.sum(lp * pi,dim=1)/ pi.shape[1]
            loss = loss - entropy*alpha

        if reduce:
            return torch.mean(loss)
        else:
            return loss

    def loss( pi_mu_sigma, y, reduce=True, entropy_reg=False, loss_type="EM", alpha=2):
        if loss_type == "EM":
            return GMMOutput.emloss(pi_mu_sigma ,y ,reduce, entropy_reg, alpha=alpha)
        elif loss_type == "MAP" :
            return GMMOutput.maploss (pi_mu_sigma, y ,reduce, entropy_reg, alpha=alpha)
        else :
            raise Exception("Loss not implemented yet")

    def forward(self, X_train):
        return None

class GMM(GMMOutput):
    def __init__(self, n_components, pre = True, dim =1):
        super(GMM, self).__init__(n_components)

        self.pis = torch.nn.parameter.Parameter(
            torch.zeros( (dim, self.components) ) )

        self.mus = torch.nn.Parameter(
            torch.randn( dim, n_components )*2 )

        self.sigmas = torch.nn.Parameter(
            torch.randn( (dim, self.components) ) ** 2  + 1 )

        self.pre = pre

    def forward(self, X_train):
        pi = F.softmax( self.pis,dim=1)
        mu = self.mus
        sigma = torch.nn.ELU()(self.sigmas) + 1.00001

        return pi, mu, sigma

    def fit(self, scm, features ,lr = 1e-3, loss_type="EM",
                batch=248, epochs=2000,entropy_reg=False,
                m_step_iter = 10, alpha=2):
        #llp = []

        if self.pre :
            km = MiniBatchKMeans(self.components)
            km.fit(scm._sample(batch)[features])
            cls = km.cluster_centers_

            self.mus = torch.nn.Parameter(
                torch.tensor(cls.T,dtype=torch.float32)
                )


        optim = torch.optim.AdamW(
            [self.pis,self.mus, self.sigmas], lr=lr)

        lossap = []

        if loss_type == "MAP" :
            m_step_iter = 1

        for i in range(epochs):

            #llp.append( self.pis )

            smps = scm._sample(batch)
            X_train = smps[features]

            for _ in range(m_step_iter):

                pi_mu_sigma = self.forward(X_train)
                #llp.append( pi_mu_sigma[1].detach().numpy().ravel() )

                energy = GMMOutput.loss( pi_mu_sigma,
                        X_train, entropy_reg=entropy_reg, loss_type=loss_type, alpha=alpha)

                optim.zero_grad()

                energy.backward()
                optim.step()

                lossap.append(energy.detach().item())

        return lossap #, llp

class MDN(GMMOutput):
    def __init__(self, n_hidden, n_components, act = torch.nn.LeakyReLU() ):
        super(MDN,self).__init__(n_components)

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

    def predict(self, X_train):

        X_train = X_train
        pi, mu, _ = self.forward(X_train)

        return torch.einsum("ij,ij->i",pi,mu).detach().numpy()

    def fit(self, scm, features="X", labels="Y", lr=1e-3, batch=248, epoch = 300, loss_type="EM", m_step_iter = 10,alpha=2, reg=False):

        optim = torch.optim.AdamW(self.parameters(), lr=lr)

        lossap = []

        if loss_type == "MAP":
            m_step_iter = 1

        for i in range(epoch):
            smps = scm._sample(batch)
            X_train = smps[features]
            Y_train = smps[labels]

            for _ in range(m_step_iter):
                y_h = self.forward(X_train)
                energy = GMMOutput.loss(y_h, Y_train, reduce=True, loss_type=loss_type, entropy_reg = reg,alpha=alpha)
                optim.zero_grad()
                energy.backward()
               # torch.nn.utils.clip_grad_norm_(self.parameters(), )
                optim.step()

                lossap.append(energy.detach().item())

        return lossap
