import argparse

from mhmm import plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=1,
                        help='''Number of recent outputs to
                                plot, starting with the most
                                recent''',
                        type=int)
    parser.add_argument('--outputdir', default='output',
                        help='''The directory where to find
                                the output files''')
    parser.add_argument('--save', default=0,
                        type=int,
                        help='''Whether to save or not''')

    opt = parser.parse_args()

    plot.plot_latest(vars(opt))


