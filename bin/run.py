import argparse
import json

from mhmm import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='direct',
                        help='''The algorithm to use when training. Can be
                             one of either "viterbi", "map" or "direct".''')
    parser.add_argument('--show', default=0,
                        help='''Whether to show the plots immediately
                                or not (0 or 1).''', 
                        type=int)
    parser.add_argument('--plotdata', default=0,
                        help='''Whether to show a plot of the dummy data
                                or not (0 or 1).''', 
                        type=int)
    parser.add_argument('--data', default='dummy',
                        help='''Which data to use. Can be either "fake" or
                             "dummy".''', 
                        type=str)
    parser.add_argument('--exp', default='fake-data',
                        help='''Which experiment to use.
                             ''', 
                        type=str)
    
    opt = vars(parser.parse_args())

    with open(f'experiments/{opt["exp"]}.json') as json_file:
        experiment_args = json.load(json_file)

    _ = opt.update(experiment_args)

    main.main(opt)
