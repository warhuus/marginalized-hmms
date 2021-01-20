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

    opt = parser.parse_args()

    plot.plot_latest(opt)


