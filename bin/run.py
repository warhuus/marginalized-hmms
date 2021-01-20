import argparse

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
    opt = parser.parse_args()

    main.main(opt)
