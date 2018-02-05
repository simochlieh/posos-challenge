import argparse


parser = argparse.ArgumentParser(description='This description is shown when -h or --help are passed as arguments.')
parser.add_argument('--required_0',
                    type=int,
                    choices=[1, 2, 3],
                    required=True,
                    help='This is a required parameter')

parser.add_argument('--multi_integer',
                    type=int,
                    default=[1, 2],
                    nargs='+',
                    help='This is a multi integer parameter. You have to provide +(at least one) integers.')

parser.add_argument('-f',
                    '--foo',
                    action='store_true',
                    help='This is a flag parameter it can be set or not.')


def main(_args):
    print(_args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
