from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import datetime
from configuration import args
import matplotlib.pyplot as plt
from matplotlib import gridspec

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 12))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 1))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    timestamp = args.save_bayesian_path +"/step_"+ str(steps) + ".pdf"
    plt.savefig(timestamp)

def plot_xy(x, y):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'The relationship between x and y',
        fontdict={'size': 30}
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    axis.plot(x, y, linewidth=3, label='Ground Truth')
    filename = args.save_bayesian_path + "/x_y" + ".pdf"
    plt.savefig(filename)

def plot_training(epoch_iter, loss, filename):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'The relationship between x and y',
        fontdict={'size': 30}
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    # axis.set_ylim((0, 0.01))
    axis.plot(epoch_iter, loss, linewidth=3, label='Training Loss')
    plt.savefig(filename)


# test for a simple function
# def target(x1, x2):
#     return np.exp(-(x1 - 2)**2) + np.exp(-(x2 - 6)**2/10) + 1/ (x1**2 + 1)
#
#
# x1 = np.linspace(0.1, 1, 100).reshape(-1, 1)
# x2 = np.linspace(-2, 1, 100).reshape(-1, 1)
# x = [x1,x2]
#
# y = target(x1,x2)
#
# optimizer = BayesianOptimization(target, {'x1': (-2, 10), 'x2': (-2, 10), }, random_state=27)


# def target(x):
#     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
#
# x = np.linspace(0.1, 1, 10000).reshape(-1, 1)
# y = target(x)
#
# plt.plot(x, y)
# optimizer = BayesianOptimization(target, {'x': (0.1, 1)}, random_state=27)
#
# optimizer.maximize(init_points=2, n_iter=0, kappa=5)
#
# plot_gp(optimizer, x, y)
#
# optimizer.maximize(init_points=0, n_iter=5, kappa=5)
# plot_gp(optimizer, x, y)