import numpy as np
from mhmm import main, plot

opt = {
    "data": "dummy",
    "N": 100,
    "D": 2,
    "K": 2,
    "N_iter": 1000,
    "reps": 3,
    "seed": 0,
    "data_seed": 4,
    "lrate": 0.05,
    "show": 0,
    "N_seq": 10,
    "optimizer": "LBFGS",
    "algo": "direct"
}

lrates = [0.05]
data_seeds = np.random.randint(1000, size=3)
seeds = data_seeds + 5
data_seeds_seeds = [(909, 914), (161, 166), (128, 133)]
Ns_Nseqs = [(100, 50)]
Ds_Ks = [(5, 2), (20, 3), (20, 5), (50, 5)]
algos_optimizers = [*[('direct', optim) for optim in ['SGD', 'adam', 'LBFGS']], ('viterbi', 'adam')]
print(data_seeds_seeds)
print([len(l) for l in [lrates, data_seeds_seeds, Ns_Nseqs, Ds_Ks, algos_optimizers]])


from itertools import product
import os

for i, (lrate, data_seed_seed, N_Nseq, D_K, algo_optimizer) in enumerate(product(lrates, data_seeds_seeds, Ns_Nseqs, Ds_Ks, algos_optimizers)):
  
  if i < 10:
   continue

  opt["data_seed"], _ = data_seed_seed
  _, opt["seed"] = data_seed_seed
  opt["N"], _ = N_Nseq
  _, opt["N_seq"] = N_Nseq
  opt["D"], _ = D_K
  _, opt["K"] = D_K
  opt["algo"], _ = algo_optimizer
  _, opt["optimizer"] = algo_optimizer
  N_iter = opt["D"] * opt["K"] * 10
  opt['N_iter'] = min(max(N_iter, 250), 2000)
  opt['lrate'] = 0.0005
  opt['where'] = 'local'

  print(i)
  print(opt)
  
  main.main(opt, dir_=r'C:\Users\chswa\OneDrive - Ørsted\DTU\semester_2\hmm_specialkursus\mhmm\output\05_08_expers')
  plot.plot_latest({'num': 1, 'outputdir': r'C:\Users\chswa\OneDrive - Ørsted\DTU\semester_2\hmm_specialkursus\mhmm\output\05_08_expers', 'save': 0})

  print('\n\n')