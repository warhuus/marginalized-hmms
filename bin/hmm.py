#%% Import packages etc.
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import cluster
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Helper functions
def drawnow():
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

def lsum(x, dim=0):
    maxx, _ = x.max(dim=dim, keepdim=True)
    return maxx + torch.log(torch.sum(torch.exp(x-maxx), dim=dim, keepdim=True))

def lsum_numpy(x, dim=0):
    maxx = x.max(axis=dim, keepdims=True)
    return maxx + np.log(np.sum(np.exp(x-maxx), axis=dim, keepdims=True))

#%% Log likelihood function
def logL(x, m, v, s):
    num_observations = x.shape[1]
    K = m.shape[1]
    e = torch.zeros(num_observations, K, device=device)
    for k in range(K):
        mm = m[:, k]
        ss = s[:, :, k]
        vv = v[:, k]
        ld = 0.5*torch.logdet(torch.matmul(ss.t(), ss) + torch.diag(vv**2))
        sx = torch.matmul(ss, x - mm[:, None])
        vx = (x - mm[:, None]) * vv[:, None]
        e[:,k] = - 0.5*torch.sum(sx**2, dim=0) - 0.5*torch.sum(vx**2, dim=0) + ld
    return e

#%% Settings
num_observations = 250
num_dimensions = 10
num_states = 5
rank_covariance = 0

#%% Create a simple data set
state_length = 10
variance = 0.1

def create_data(num_observations, num_states, num_dimensions):
    m = np.ones((num_states, num_dimensions)) * np.arange(num_states)[:,None]
    c = np.floor_divide(np.arange(num_observations), state_length) % num_states
    s = np.tile(np.eye(num_dimensions),(num_states,1,1)) * np.sqrt(variance)
    
    x = np.zeros((num_dimensions, num_observations))
    for n in range(num_observations):       
        i = c[n]
        x[:, n] = np.random.multivariate_normal(m[i], s[i])
    
    return torch.tensor(x, dtype=torch.float32)

X0 = create_data(num_observations, num_states, num_dimensions)
plt.figure('Data')
plt.clf()
plt.imshow(X0.numpy())
drawnow()

#%% Model parameters
par = {'requires_grad':True, 'dtype':torch.float32, 'device':device}
ulog_T = torch.randn((num_states,num_states), **par)
ulog_t0 = torch.randn(num_states, **par)
M = torch.randn((num_dimensions,num_states), **par)
V = torch.ones((num_dimensions,num_states), **par)
S = torch.randn((rank_covariance,num_dimensions,num_states), **par)

#%% Fit model
X0 = X0.to(device)

# Number of iterations
R = 1000
n_plot = 50 # Plot every n_plot iterations

# Minibatch log-likelihood
Lr = np.repeat(np.nan, R)

# Optimizer
optimizer = torch.optim.Adam([ulog_T, ulog_t0, M, V, S], lr=0.05)

# Minibatch size
length_snip = 10

for r in range(R):
    i0 = np.random.randint(num_observations-length_snip+1)
    X = X0[:,i0:i0+length_snip]
    
    log_T = ulog_T - lsum(ulog_T, dim=0)
    log_t0 = ulog_t0 - lsum(ulog_t0)
    E = logL(X, M, V, S)
    
    log_p = log_t0+E[0]    
    for n in range(1,length_snip):
        log_p = torch.squeeze(lsum(log_p + log_T, dim=1))+E[n]
    L = -lsum(log_p)
    
    Lr[r] = L.detach().cpu().numpy()      
    
    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    # Plot every n_plot iterations
    if ((r+1) % n_plot)==0:
        
        plt.figure('Objective').clf()
        plt.plot(Lr)            
        drawnow()
        
        plt.figure('Parameters').clf()
        S_numpy = S.detach().cpu().numpy()
        V_numpy = V.detach().cpu().numpy()
        covariance_matrix = np.empty((num_states,num_dimensions,num_dimensions))
        for i in range(num_states):
            plt.subplot(2,num_states,i+1)
            covariance_matrix[i,:,:] = np.linalg.inv(np.matmul(S_numpy[:,:,i].T, S_numpy[:,:,i])+np.diag(V_numpy[:,i]**2))
            plt.imshow(covariance_matrix[i,:,:])
            plt.colorbar()

        plt.subplot(223)
        plt.imshow(M.detach().cpu().numpy())
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(np.exp(log_T.detach().cpu().numpy()))
        plt.clim(0,1)
        plt.colorbar()
        drawnow()
        
        plt.figure('State probability')    
        plt.clf()
        log_T = ulog_T - lsum(ulog_T, dim=1)
        log_t0 = ulog_t0 - lsum(ulog_t0)
        E = logL(X0.to(device), M, V, S)

        log_p = log_t0
        log_p_n = np.zeros((num_observations, num_states))
        log_p_n[0] = log_p.detach().cpu().numpy() + E[0].detach().cpu().numpy()
        for n in range(1,num_observations):
            log_p = torch.squeeze(lsum(log_p + log_T, dim=1))+E[n]
            log_p_n[n] = log_p.detach().cpu().numpy()        

        log_p = torch.zeros(num_states, device=device)
        log_p_b = np.zeros((num_observations, num_states))        
        log_p_b[-1] = log_p.detach().cpu().numpy()
        for n in reversed(range(num_observations-1)):
            log_p = torch.squeeze(lsum(log_p + E[n+1] + log_T.t(), dim=1))
            log_p_b[n] = log_p.detach().cpu().numpy()
            
        log_p_t = log_p_n+log_p_b
        p = np.exp(log_p_t - lsum_numpy(log_p_t, dim=1))
        plt.imshow(p.T, aspect='auto', interpolation='none')
        drawnow()
