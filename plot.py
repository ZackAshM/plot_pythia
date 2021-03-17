# plot pythia data histogram - default: energy

import argparse

import numpy as np
from numpy import logical_or as OR
from numpy import logical_and as AND

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# command line arguments: data name and output name
parser = argparse.ArgumentParser(
    description='Plots the given pythia data file.')
parser.add_argument('DATA_NAME', type=str,
                    help='str; the name of the data file')
parser.add_argument('-s', '--save_name', default='', type=str,
                    help='str; the name of the saved png')
args = parser.parse_args()

# parse args
DATAFILE = args.DATA_NAME
SAVEFILE = args.save_name

# read pythia data
newdata = []
with open(DATAFILE) as rawdata:
    lines = rawdata.readlines()
    for line in lines:
        ls = line.split()
        if len(ls) <= 14:
            if ls[0] == 'I':
                col_names = ls
            else:
                if '=' not in ls[0] and 'P' not in ls[0]:
                    newdata.append(ls)

# convert data to pandas dataframe
data = pd.DataFrame(newdata, columns=col_names)
data = data.astype(float)

# filter relevant data
stable = data['K(I,1)'] == 1
data = data[stable]  # only select stable particles

photons = data['K(I,2)'] == 22
electrons = data['K(I,2)'] == 11
protons = data['K(I,2)'] == 2212
pions = np.abs(data['K(I,2)']) == 211
kaons = OR(OR(OR(data['K(I,2)'] == 311, data['K(I,2)'] == 321), 
              data['K(I,2)'] == 310), data['K(I,2)'] == 130)

energy = data['P(I,4)']
px = data['P(I,1)']
py = data['P(I,2)']
pz = data['P(I,3)']

# analyze pT (transverse momentum) = cosh(eta) / E
# eta as defined in third formula at: https://en.wikipedia.org/wiki/Pseudorapidity
def pT(E, px, py, pz, pl = None):
    if pl is None:
        pl = pz
    eta = np.arctanh(pl / np.sqrt(px**2.0 + py**2.0 + pz**2.0))
    return eta, E / np.cosh(eta)

eta, transverse_mom = pT(energy, px, py, pz)

# only interested in eta < 4.5 (low scatter)
data['eta'] = eta
eta_max = np.abs(eta) < 4.5 
data['pT'] = transverse_mom
data_pT = data['pT']

# save npy
npdata = data.to_numpy()
np.save(SAVEFILE+'_data.npy',npdata)

# plot data
fig, ax = plt.subplots(figsize=[15, 10])

# need to change if analyzing another quantity
ax.set(xlabel='Energy (GeV)', ylabel='Counts',
       title='20 GeV e- on 100 GeV/nuc Au')

# plot for each particle listed
particles = [photons, pions, kaons, protons, electrons]
labels = ['photons', 'pions', 'kaons', 'protons', 'electrons']
colors = ['black', 'purple', 'blue', 'red', 'green']

for particle, label, c in zip(particles, labels, colors):
    plt.hist(x=energy[particle], bins=20, alpha=0.7, histtype='step', log=True, color=c, lw=2, label=label)
    ax.axvline(np.quantile(energy[particle], 0.5), linestyle='dashed', c=c, lw=2, label=label+' 50% quantile')
plt.grid(axis='y', alpha=0.75)
plt.legend()

# save the plot
plt.savefig(SAVEFILE+'_energy.png')

# plt.show()
