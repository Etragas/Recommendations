import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import pickle
from collections import defaultdict
widths = [3, 10, 30, 100, 300, 1000, 3000]

print "Plotting results..."
#with open('results.pkl') as f:
#      all_results = pickle.load(f)

#for key in all_results[0]:
#    plot_traces_and_mean(all_results, key)

rc('font',**{'family':'serif'})
fig = plt.figure(0); fig.clf()


#X = all_results[0]["iterations"]
ax = fig.add_subplot(211)
plt.locator_params(nbins=4, axis='y')
#final_train = [all_results[i]["train_likelihood"][-1] for i, w in enumerate(widths)]
#final_tests = [all_results[i]["tests_likelihood"][-1] for i, w in enumerate(widths)]
X = np.arange(10,110,10)
final_train = [1.09567005,1.07211951,1.06343562,1.05429676,1.0513782,1.04671063,1.04445803,1.0439747,1.04408041,1.04426326]
final_1m = [1.0292421,1.00841532,1.00813826,1.00879561,1.00827666,1.00866983,1.0081898,1.00840027,1.00850809,1.00850086]

plt.plot(X,final_train, label='ML-100K')
plt.plot(X,final_1m, label='ML-1M')
plt.legend(frameon=False,prop={'size':'12'})
#plt.plot(final_tests[1:], label='Test likelihood')
#plt.xticks(X)#range(len(widths[1:])), widths[1:])
#ax.legend(numpoints=1, loc=4, frameon=False, prop={'size':'12'})
ax.set_ylabel('RMSE On Test')
ax.set_xlabel('Percent of Online data available')
#

#    ax = fig.add_subplot(212)
#    plt.locator_params(nbins=4, axis='y')
#    final_marg = [all_results[i]["marg_likelihood"][-1] for i, w in enumerate(widths)]
#    plt.plot(final_marg[1:], 'r', label='Marginal likelihood estimate')
#    ax.legend(numpoints=1, loc=4, frameon=False, prop={'size':'12'})
#    ax.set_ylabel('Marginal likelihood')
#    ax.set_xlabel('Number of hidden units')
#    plt.xticks(range(len(widths[1:])), widths[1:])
#    #low, high = ax.get_ylim()
#    #ax.set_ylim([0, high])
#    # plt.show()
#    fig.set_size_inches((5,3.5))
#    #ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
plt.savefig('vary_widths.pdf', pad_inches=0.05, bbox_inches='tight')
plt.savefig('vary_widths.png', pad_inches=0.05, bbox_inches='tight')

    # # Nice plots for paper.
    # rc('font',**{'family':'serif'})
    # fig = plt.figure(0); fig.clf()
    # X = all_results[0]["iterations"]
    # ax = fig.add_subplot(212)
    # for i, w in enumerate(widths):
    #     plt.plot(X, all_results[i]["marg_likelihood"],
    #              label='{0} hidden units'.format(w))
    # ax.legend(numpoints=1, loc=3, frameon=False, prop={'size':'12'})
    # ax.set_ylabel('Marginal likelihood')

    # ax = fig.add_subplot(211)
    # for i, w in enumerate(widths):
    #     plt.plot(X, all_results[i]["tests_likelihood"])
    # ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    # ax.set_ylabel('Test Likelihood')
    # ax.set_xlabel('Training iteration')
    # #low, high = ax.get_ylim()
    # #ax.set_ylim([0, high])
    # fig.set_size_inches((5,3.5))
    # #ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    # plt.savefig('vary_widths.pdf', pad_inches=0.05, bbox_inches='tight')
