import argparse
import json

from mhmm import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='direct',
                        help='''The algorithm to use when training. Can be
                             one of either "viterbi", "map" or "direct".''')
    parser.add_argument('--exp', default='fake-data',
                        help='''Which experiment to use.
                             ''', 
                        type=str)
    parser.add_argument('--lrate', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)

    opt = vars(parser.parse_args())

    with open(f'experiments/{opt["exp"]}.json') as json_file:
        experiment_args = json.load(json_file)

    opt.update(experiment_args)

    print(opt, '\n\n')

    main.main(opt)
