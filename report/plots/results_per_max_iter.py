import numpy as np
import pandas as pd

data = pd.read_csv('results/SNRvsPe.csv.gz')

# mark all words that took 20 iterations as unsolvable
data[data['iterations'] == 20] = np.infty

# mark words solved with less than a certain
# number of message passing iterations
max_iters = [1, 3, 5, 8, 11, 15, 18, 20]
for max_iter in max_iters:
    data[max_iter] = (data['iterations'] > max_iter).astype(int)

del data['time per word']
del data['errors']
del data['iterations']

data = pd.melt(data,
               id_vars = ['n', 'rate', 'SNR'],
               var_name = 'max_iter',
               value_name = 'errors')

# remove dummy values, convert back to integer
data = data[data['n'] != np.infty]
data['n'] = data['n'].astype(int)

ns = data['n'].unique()
rates = data['rate'].unique()
SNRs = data['SNR'].unique()

# create index, based on possible combinations
new_index = pd.MultiIndex.from_product(
    [ns, rates, SNRs, max_iters],
    names=['n', 'rate', 'SNR', 'max_iters'])

# compute error probability per max_iter threshold
data = data.groupby(['n', 'rate',  'SNR', 'max_iter']).mean()

# fill missing values with following one
# (0 if any is available) as we want
data = data.reindex(new_index)\
           .fillna(method='pad')

data.reset_index().to_csv('results/intermediate_plot_computation.csv.gz',
                          compression='gzip',
                          index=None)
