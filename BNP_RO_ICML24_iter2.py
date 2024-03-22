import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression







#############
# Functions #
#############

def stick_breaking_sampling(T_steps, n, alpha):
    # Sample T_steps + 1 stick-breaking weights
    tmp = np.random.beta(1, alpha + n, size = T_steps).cumprod()
    
    return np.append(tmp, 1-sum(tmp))



def dirichlet_multinomial_sampling(T_steps, n, alpha):
    # Sample T_steps + 1 Dirichlet weights
    shape = (alpha + n)/(T_steps + 1)
    tmp = np.random.gamma(shape, 1, size = T_steps + 1)
    
    return tmp / tmp.sum()



def atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0 = 0):
    # Sample atoms according to the DP predictive
    if loss_fun != 'gaussian_loc_lik':
        d = len(data[0,:])
    p = alpha / (alpha + n)
    atoms = []
    
    for j in range(1, T_steps+2):
        if np.random.random() < p:
            if loss_fun == 'gaussian_loc_lik':
                atoms.append(np.random.normal(mn_0, 1))
            elif loss_fun == 'squared':
                atoms.append(np.random.normal(0, 1, d))
            elif loss_fun == 'logistic':
                y = np.random.choice([-1, 1])
                X = np.random.normal(0,1,d-1)
                dt = np.append(y, X)
                atoms.append(dt)
        else:
            if loss_fun == 'gaussian_loc_lik':
                atoms.append(data[np.random.randint(0,n)])
            elif loss_fun == 'squared':
                atoms.append(data[np.random.randint(0,n),:])
            elif loss_fun == 'logistic':
                atoms.append(data[np.random.randint(0,n),:])
    return atoms
        


def approx_criterion(N_mc, T_steps, approx_type, data, alpha, loss_fun, mn_0 = 0):
    # Wrap weights and atoms together
    if loss_fun == 'gaussian_loc_lik':
        n = len(data)
    elif loss_fun == 'squared':
        n = len(data[:,0])
    elif loss_fun == 'squared':
        n = len(data[:,0])
        
    if approx_type == 'stick_breaking':    
        return [
            {'weights' : stick_breaking_sampling(T_steps, n, alpha),
             'atoms' : atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0)
            }
            for i in range(0, N_mc)
        ]
    elif approx_type == 'dirichlet_multinomial':
        return [
            {'weights' : dirichlet_multinomial_sampling(T_steps, n, alpha),
             'atoms' : atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0)
            }
            for i in range(0, N_mc)
        ]



def h(loss_fun, theta, xi):
    # Compute loss_fun at single data point xi and at parameter theta.
    # For regression loss, first column of xi is y, all other columns are X
    
    # 1-D Gaussian location MLE loss
    if loss_fun == 'gaussian_loc_lik':
        h = 1.e-3*np.square(xi - theta, dtype = np.float64)
    
    # Linear regression loss, both y and X are real
    elif loss_fun == 'squared':
        reg_term = xi[0] - np.matmul(xi[1:], theta)
        h = 1.e-3*np.square(reg_term, dtype = np.float64)

    # Logistic regression loss, y is in {-1,1}, X is real
    elif loss_fun == 'logistic':
        clas_term = xi[0]*np.matmul(xi[1:], theta)
        h = 1.e-3*np.log(1 + np.exp(-clas_term, dtype = np.float64))
    
    return h



def grad(loss_fun, theta, xi):
    # Compute gradient wrt theta of loss_fun at single data point xi
    # For regression loss, first column of xi is y, all other columns are X
    
    # 1-D Gaussian location MLE loss
    if loss_fun == 'gaussian_loc_lik':
        g = -2*1.e-3*(xi - theta)
    
    # Linear regression loss, both y and X are real
    elif loss_fun == 'squared':
        reg_term = xi[0] - np.matmul(xi[1:], theta)
        g = 1.e-3*np.array([-2*(reg_term)*xi_j for xi_j in xi[1:]], dtype = np.float64)
    
    # Logistic regression loss, y is in {-1,1}, X is real
    elif loss_fun == 'logistic':
        clas_term = xi[0]*np.matmul(xi[1:], theta)
        exp_term = np.exp(-clas_term, dtype = np.float64)
        g = -1.e-3*np.array([xi[0]*xi_j for xi_j in xi[1:]])*exp_term/(1 + exp_term)

            
    return g



def phi(t, beta):
    # Compute second-order utility phi at t and ambiguity index beta
    clip = min(t/beta, 700) ## avoid overflow issues with np.exp
    return (beta * np.exp(clip)) - beta


    
def phi_inv(t, beta):
    # Compute inverse of second-order utility phi at t and ambiguity index beta
    return beta * np.log((t / beta) + 1)


    
def phi_prime(t, beta):
    # Compute derivative of second-order utility phi at t and ambiguity index beta
    clip = min(t/beta, 700) ## avoid overflow issues with np.exp
    return np.exp(clip)


    
def criterion_value(loss_fun, theta, beta, criterion):
    # Compute criterion value, applying inverse transformation phi_inv
    # to make criterion values comparable across beta values
    N_mc = len(criterion)
    T_steps = len(criterion[0]['weights'])
    
    return phi_inv(np.array([
        phi(np.array([h(loss_fun, theta, criterion[i]['atoms'][k]) * criterion[i]['weights'][k] for k in range(0, T_steps)]).sum(), beta)
        for i in range(0,N_mc)
    ]).mean(), beta)



def SGD_alternative(loss_fun, theta_0, beta, criterion, n_passes, step_size0):
    # Perform SGD updates based on whole MC samples, so to reduce
    # computation at each iteration (the loss function is evaluated only at one sample).
    # Each MC sample is used at each pass (sampled without replacement), and the
    # procedure is repeated n_passes times
    
    N = len(criterion)
    theta_path = [theta_0]
    values_path = [criterion_value(loss_fun, theta_0, beta, criterion)]
    
    iteration = 1
    for t in range(1, n_passes + 1):
        indexes = [idx for idx in range(0, N)]
        for n in range(1, N + 1):
            idx = np.random.choice(indexes)
            indexes = [i for i in indexes if i != idx]
            theta_tm1 = theta_path[-1]
            eta = step_size0/(100 + np.sqrt(iteration))
            
            atoms = criterion[idx]['atoms']
            weights = criterion[idx]['weights']
            loss_vals = np.array([h(loss_fun, theta_tm1, xi) for xi in atoms])
            grad_vals = np.array([grad(loss_fun, theta_tm1, xi) for xi in atoms])
            
            theta_t = theta_tm1 - eta * phi_prime(loss_vals.dot(weights), beta) * np.matmul(weights, grad_vals)
            
            theta_path.append(theta_t)
            values_path.append(criterion_value(loss_fun, theta_t, beta, criterion))
            
            iteration += 1
    
    return theta_path, values_path



def oos_performance(loss_fun, theta, tst_sample):
    # Compute out-of-sample performance at theta on a test_sample
    return np.array([h(loss_fun, theta, xi) for xi in tst_sample]).mean()

    











###########################################
# Linear Regression Simulation Experiment #
###########################################

# Data parameters
dim = 90
active_params = 5
true_coefs = np.append(np.ones(active_params), np.zeros(dim-active_params))
means = np.zeros(dim)
covariance = 0.3*np.ones(dim) + (1-0.3)*np.eye(dim)
n_train = 100
n_test = 5000
n_tot = n_train + n_test

# Criterion parameters
loss_fun = 'squared'
beta_grid = [1, 'inf']
alpha_grid = [1, 2, 5, 10]
approx_type = 'dirichlet_multinomial'
N_mc = 300
T_steps = 50

# SGD parameters
n_steps = 4000
step_size0 = 50
n_passes = int(np.ceil(n_steps/N_mc))

np.random.seed(1234)

BNP_oos = {}
OLS_oos = {}
BNP_theta = {}
OLS_theta = {}

n_sims = 200
theta_0 = np.zeros(dim)
for sim in range(1, n_sims + 1):
    data = np.random.multivariate_normal(means, covariance, n_tot)
    y = np.sum(data[:,:active_params], axis = 1) + np.random.normal(0, 0.5, n_tot)

    data = np.column_stack((y, data))
    trn_sample = data[0:n_train,:]
    tst_sample = data[n_train:,:]

    theta_ols = np.linalg.inv(trn_sample[:,1:].transpose()@trn_sample[:,1:])@trn_sample[:,1:].transpose()@trn_sample[:,0]
    OLS_oos[(n_train, sim)] = oos_performance(loss_fun = loss_fun, theta = theta_ols, tst_sample = tst_sample)
    OLS_theta[(n_train, sim)] = theta_ols


    for alpha in alpha_grid:

        a = approx_criterion(N_mc = N_mc, T_steps = T_steps, approx_type = approx_type, data = trn_sample, alpha = alpha, loss_fun = loss_fun)
        theta_ridge = np.linalg.inv(trn_sample[:,1:].transpose()@trn_sample[:,1:] + (alpha/n_train)*np.eye(dim))@trn_sample[:,1:].transpose()@trn_sample[:,0]

        for beta in beta_grid:
            if beta == 'inf':  
                BNP_theta[(n_train, sim, alpha, beta)] = theta_ridge
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_ridge, tst_sample = tst_sample)   
            else:
                theta_path, values_path = SGD_alternative(loss_fun = loss_fun,
                                                          theta_0 = theta_0, beta = beta, criterion = a,
                                                          n_passes = n_passes, step_size0 = step_size0)
                BNP_theta[(n_train, sim, alpha, beta)] = theta_path[-1]
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_path[-1], tst_sample = tst_sample)
                

# Store results in dataframe

df1 = pd.DataFrame(columns = ['n_train', 'sim', 'alpha', 'beta', 'oos_performance', 'theta_norm', 'theta_dist_truth'])

indexes = []
for sim in range(1, n_sims + 1):
    for alpha in alpha_grid:
        indexes.append([n_train, sim, alpha, 'ols'])
        for beta in beta_grid:
            indexes.append([n_train, sim, alpha, beta])

df1[['n_train', 'sim', 'alpha', 'beta']] = indexes


for sim in range(1, n_sims + 1):
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'oos_performance'] = np.sqrt(OLS_oos[(n_train, sim)])
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_norm'] = np.sqrt(np.square(OLS_theta[(n_train, sim)]).sum())
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_dist_truth'] = np.sqrt(np.square(OLS_theta[(n_train, sim)] - true_coefs).sum())
    for alpha in alpha_grid:
        for beta in beta_grid:
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'oos_performance'] = np.sqrt(BNP_oos[(n_train, sim, alpha, beta)])
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'theta_norm'] = np.sqrt(np.square(BNP_theta[(n_train, sim, alpha, beta)]).sum())
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'theta_dist_truth'] = np.sqrt(np.square(BNP_theta[(n_train, sim, alpha, beta)] - true_coefs).sum())


# Plot of Results (Figure 2)

labels = {'oos_performance': 'Test RMSE', 'theta_dist_truth': r'L2 Dist. from Truth ($\hat\theta$)', 'theta_norm': r'L2 Norm ($\hat\theta$)'}
alpha_grid = [1, 2, 5, 10]

df1.loc[df1['beta'] == 1, 'beta'] = 'A. Averse'
df1.loc[df1['beta'] == 'inf', 'beta'] = 'A. Neutral'
df1.loc[df1['beta'] == 'ols', 'beta'] = 'OLS'

fig, axs = plt.subplots(nrows=len(labels), ncols=len(alpha_grid), figsize=(len(alpha_grid) * 5, len(labels) * 3), sharex='col', sharey='row')

for j, metric in enumerate(labels.keys()):
    axs[j, 0].set_ylabel(labels[metric], fontsize=16)

for j, metric in enumerate(labels.keys()):
    for i, alpha in enumerate(alpha_grid):
        tmp = df1.loc[df1['alpha'] == alpha, ]
        tmp_mean = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].mean().reset_index()
        tmp_std = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].std().reset_index()
        x = np.arange(3)

        ax = axs[j, i]
        bar1 = ax.bar(x - 0.2, tmp_mean[metric], width=0.4, label='Mean', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp_mean['beta'], fontsize=14)  
        ax.tick_params(axis='y', labelsize=14, labelcolor = '#1f77b4')  

        if j == 0:
            ax.set_title(fr'$\alpha = {alpha}$', fontsize=20)

        ax2 = ax.twinx()
        bar2 = ax2.bar(x + 0.2, tmp_std[metric], width=0.4, label='St. Dev', color = '#ff7f0e', zorder=2)
        ax2.tick_params(axis='y', labelsize=14, labelcolor = '#ff7f0e') 

        if i != len(alpha_grid) - 1:
            ax2.set_yticks([])
            ax2.yaxis.tick_right()

        ax2.grid(False)

        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)

        ax.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
        ax.grid(axis='y', linestyle='-', alpha=0.5, zorder=1)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))

handles = [bar1, bar2]
labels = ['Mean', 'St. Dev']
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)

plt.tight_layout()
plt.show()






    





##################################################
# Gaussian Mean Estimation Simulation Experiment #
##################################################

# Data parameters
n_sims = 100
n_train = 10
n_test = 5000
n_tot = n_train + n_test
n_out = 3 # number of outlier observations
mn_true, mn_out = 0, 5
s = 1

# Criterion parameters
loss_fun = 'gaussian_loc_lik'
beta_grid = [1, 'inf']
alpha_grid = [1, 2, 5, 10]
approx_type = 'dirichlet_multinomial'
N_mc = 300
T_steps = 50
mn_0 = (n_train * mn_true + n_out * mn_out) / (n_train + n_out)


# SGD parameters
n_steps = 8000
step_size0 = 30
n_passes = int(np.ceil(n_steps/N_mc))


np.random.seed(12345)

BNP_oos = {}
MLE_oos = {}
BNP_theta = {}
MLE_theta = {}


for sim in range(1, n_sims + 1):
    data = np.random.normal(mn_true, s, n_tot)

    trn_sample = data[0:n_train]
    trn_sample = np.concatenate((trn_sample, np.random.normal(mn_out, s, n_out)))
    tst_sample = data[n_train:]

    theta_MLE = trn_sample.mean()
    MLE_oos[(n_train, sim)] = oos_performance(loss_fun = loss_fun, theta = theta_MLE, tst_sample = tst_sample)
    MLE_theta[(n_train, sim)] = theta_MLE
    theta_0 = 0
    
    for alpha in alpha_grid:
        theta_a_neut = np.concatenate((trn_sample, np.array([mn_0 for i in range(0, alpha)]))).mean()            
        a = approx_criterion(N_mc = N_mc, T_steps = T_steps, approx_type = approx_type, data = trn_sample, alpha = alpha, loss_fun = loss_fun)
        for beta in beta_grid:
            if beta == 'inf':  
                BNP_theta[(n_train, sim, alpha, beta)] = theta_a_neut
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_a_neut, tst_sample = tst_sample)
            else:
                theta_path, values_path = SGD_alternative(loss_fun = loss_fun,
                                                          theta_0 = theta_0, beta = beta, criterion = a,
                                                          n_passes = n_passes, step_size0 = step_size0)
                BNP_theta[(n_train, sim, alpha, beta)] = theta_path[-1]
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_path[-1], tst_sample = tst_sample)                


# Store results in dataframe

df = pd.DataFrame(columns = ['n_train', 'sim', 'alpha', 'beta', 'oos_performance', 'theta_dist_truth'])

indexes = []
for sim in range(1, n_sims + 1):
    for alpha in alpha_grid:
        indexes.append([n_train, sim, alpha, 'MLE'])
        for beta in beta_grid:
            indexes.append([n_train, sim, alpha, beta])

df[['n_train', 'sim', 'alpha', 'beta']] = indexes

for sim in range(1, n_sims + 1):
    df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['beta'] == 'MLE'), 'oos_performance'] = MLE_oos[(n_train, sim)]
    df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['beta'] == 'MLE'), 'theta_dist_truth'] = np.abs(MLE_theta[(n_train, sim)])
    for alpha in alpha_grid:
        for beta in beta_grid:
           df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['alpha'] == alpha) & (df['beta'] == beta), 'oos_performance'] = BNP_oos[(n_train, sim, alpha, beta)]
           df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['alpha'] == alpha) & (df['beta'] == beta), 'theta_dist_truth'] = np.abs(BNP_theta[(n_train, sim, alpha, beta)])
   

# Plot of Results (Figure 3)

labels = {'oos_performance': 'Test Mean Neg. Log-Lik.', 'theta_dist_truth': r'Dist. from Truth ($\hat\theta$)'}
alpha_grid = [1, 2, 5, 10]

df.loc[df['beta'] == 1, 'beta'] = 'A. Averse'
df.loc[df['beta'] == 'inf', 'beta'] = 'A. Neutral'

fig, axs = plt.subplots(nrows=len(labels), ncols=len(alpha_grid), figsize=(len(alpha_grid) * 5, len(labels) * 3), sharex='col', sharey='row')

for j, metric in enumerate(labels.keys()):
    axs[j, 0].set_ylabel(labels[metric], fontsize=16)

for j, metric in enumerate(labels.keys()):
    for i, alpha in enumerate(alpha_grid):
        tmp = df.loc[df['alpha'] == alpha, ]
        tmp_mean = tmp.groupby('beta')[['oos_performance', 'theta_dist_truth']].mean().reset_index()
        tmp_std = tmp.groupby('beta')[['oos_performance', 'theta_dist_truth']].std().reset_index()
        x = np.arange(3)

        ax = axs[j, i]
        bar1 = ax.bar(x - 0.2, tmp_mean[metric], width=0.4, label='Mean', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp_mean['beta'], fontsize=14)  
        ax.tick_params(axis='y', labelsize=14, labelcolor = '#1f77b4')  

        if j == 0:
            ax.set_title(fr'$\alpha = {alpha}$', fontsize=20)

        ax2 = ax.twinx()
        bar2 = ax2.bar(x + 0.2, tmp_std[metric], width=0.4, label='St. Dev', color = '#ff7f0e', zorder=2)
        ax2.tick_params(axis='y', labelsize=14, labelcolor = '#ff7f0e') 

        if i != len(alpha_grid) - 1:
            ax2.set_yticks([])
            ax2.yaxis.tick_right()

        ax2.grid(False)

        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)

        ax.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
        ax.grid(axis='y', linestyle='-', alpha=0.5, zorder=1)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))

handles = [bar1, bar2]
labels = ['Mean', 'St. Dev']
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.001, 0.5), fontsize=16)

plt.tight_layout()
plt.show()









#############################################
# Logistic Regression Simulation Experiment #
#############################################

# Data parameters
dim = 90
active_params = 5
true_coefs = np.append(np.ones(active_params), np.zeros(dim-active_params))
means = np.zeros(dim)
covariance = 0.3*np.ones(dim) + (1-0.3)*np.eye(dim)
n_train = 100
n_test = 5000
n_tot = n_train + n_test

# Criterion parameters
loss_fun = 'logistic'
beta_grid = [1, 'l2']
alpha_grid = [1, 2, 5, 10]
approx_type = 'dirichlet_multinomial'
N_mc = 200
T_steps = 50

# SGD parameters
n_steps = 1400
step_size0 = 20000
n_passes = int(np.ceil(n_steps/N_mc))

np.random.seed(1234)

BNP_oos = {}
OLS_oos = {}
BNP_theta = {}
OLS_theta = {}

n_sims = 200
theta_0 = np.zeros(dim)
for sim in range(1, n_sims + 1):
    data = np.random.multivariate_normal(means, covariance, n_tot)
    pr = 1/(1+np.exp(-np.sum(data[:,:active_params], axis = 1)))
    y = np.array([np.random.choice([1, -1], size = 1, p = [pr[i], 1-pr[i]]) for i in range(0, n_tot)])

    data = np.column_stack((y, data))
    trn_sample = data[0:n_train,:]
    tst_sample = data[n_train:,:]
    
    model = LogisticRegression(penalty='none', fit_intercept=False)
    model.fit(trn_sample[:,1:], trn_sample[:,0])

    theta_ols = model.coef_.T
    
    OLS_oos[(n_train, sim)] = oos_performance(loss_fun = loss_fun, theta = theta_ols, tst_sample = tst_sample)
    OLS_theta[(n_train, sim)] = theta_ols


    for alpha in alpha_grid:
        
        model_l2 = LogisticRegression(penalty='l2', C = n_train/alpha, fit_intercept=False)
        model_l2.fit(trn_sample[:,1:], trn_sample[:,0])
        theta_ridge = model_l2.coef_.T

        a = approx_criterion(N_mc = N_mc, T_steps = T_steps, approx_type = approx_type, data = trn_sample, alpha = alpha, loss_fun = loss_fun)

        for beta in beta_grid:
            if beta == 'l2':  
                BNP_theta[(n_train, sim, alpha, beta)] = theta_ridge
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_ridge, tst_sample = tst_sample)   

            else:
                theta_path, values_path = SGD_alternative(loss_fun = loss_fun,
                                                          theta_0 = theta_0, beta = beta, criterion = a,
                                                          n_passes = n_passes, step_size0 = step_size0)

                BNP_theta[(n_train, sim, alpha, beta)] = theta_path[-1]
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun = loss_fun, theta = theta_path[-1], tst_sample = tst_sample)


# Store results in dataframe

df1 = pd.DataFrame(columns = ['n_train', 'sim', 'alpha', 'beta', 'oos_performance', 'theta_norm', 'theta_dist_truth'])

indexes = []
for sim in range(1, n_sims + 1):
    for alpha in alpha_grid:
        indexes.append([n_train, sim, alpha, 'ols'])
        for beta in beta_grid:
            indexes.append([n_train, sim, alpha, beta])

df1[['n_train', 'sim', 'alpha', 'beta']] = indexes


for sim in range(1, n_sims + 1):
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'oos_performance'] = OLS_oos[(n_train, sim)]
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_norm'] = np.sqrt(np.square(OLS_theta[(n_train, sim)]).sum())
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_dist_truth'] = np.sqrt(np.square(OLS_theta[(n_train, sim)].T - true_coefs).sum())
    for alpha in alpha_grid:
        for beta in beta_grid:
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'oos_performance'] = BNP_oos[(n_train, sim, alpha, beta)]
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'theta_norm'] = np.sqrt(np.square(BNP_theta[(n_train, sim, alpha, beta)]).sum())
           df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (df1['beta'] == beta), 'theta_dist_truth'] = np.sqrt(np.square(BNP_theta[(n_train, sim, alpha, beta)].T - true_coefs).sum())
       

# Plot of Results (Figure 4)

labels = {'oos_performance': 'Test loss', 'theta_dist_truth': r'L2 Dist. from Truth ($\hat\theta$)', 'theta_norm': r'L2 Norm ($\hat\theta$)'}
alpha_grid = [1, 2, 5, 10]

df1.loc[df1['beta'] == 1, 'beta'] = 'A. Averse'
df1.loc[df1['beta'] == 'l2', 'beta'] = 'L2 penalty'
df1.loc[df1['beta'] == 'ols', 'beta'] = 'No penalty'

fig, axs = plt.subplots(nrows=len(labels), ncols=len(alpha_grid), figsize=(len(alpha_grid) * 5, len(labels) * 3), sharex='col', sharey='row')

for j, metric in enumerate(labels.keys()):
    axs[j, 0].set_ylabel(labels[metric], fontsize=16)  # Adjust font size

for j, metric in enumerate(labels.keys()):
    for i, alpha in enumerate(alpha_grid):
        tmp = df1.loc[df1['alpha'] == alpha, ]
        tmp_mean = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].mean().reset_index()
        tmp_std = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].std().reset_index()
        x = np.arange(3)

        ax = axs[j, i]
        bar1 = ax.bar(x - 0.2, tmp_mean[metric], width=0.4, label='Mean', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp_mean['beta'], fontsize=14)  # Adjust font size
        ax.tick_params(axis='y', labelsize=14, labelcolor = '#1f77b4')  # Adjust font size of y-axis ticks

        if j == 0:
            ax.set_title(fr'$\alpha = {alpha}$', fontsize=20)  # Adjust font size

        ax2 = ax.twinx()
        bar2 = ax2.bar(x + 0.2, tmp_std[metric], width=0.4, label='St. Dev', color = '#ff7f0e', zorder=2)
        ax2.tick_params(axis='y', labelsize=14, labelcolor = '#ff7f0e')  # Adjust font size of y-axis ticks

        if i != len(alpha_grid) - 1:
            ax2.set_yticks([])
            ax2.yaxis.tick_right()

        ax2.grid(False)

        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)

        ax.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
        ax.grid(axis='y', linestyle='-', alpha=0.5, zorder=1)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.3f}'.format(x)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))

handles = [bar1, bar2]
labels = ['Mean', 'St. Dev']
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)  # Adjust font size and position

plt.tight_layout()
plt.show()