import argparse
from  common.misc_util import boolean_flag

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

boolean_flag(parser, 'domain_randomization', default=False)
parser.add_argument('--multitask', help='Multitask architecture', choices=['distral', 'tscl', 'distral_tscl'], default='cnn')
parser.add_argument('--architecture', help='Policy architecture', choices=['dnc', 'vddnc', 'ddpg', 'vdddpg', 'pathnet', 'pathmatrix', 'dnc'], default='ddpg')
parser.add_argument('--input', help='Input Type', choices=['json', 'feat', 'feason'], default='json')
parser.add_argument('--stage', help='Stage that the model is in', choices=['training', 'evaluation', 'live'], default='evaluation')

