import argparse
import omni
from  common.misc_util import boolean_flag

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

boolean_flag(parser, 'domain_randomization', default=False)
boolean_flag(parser, 'load_model', default=True)
parser.add_argument('--multitask', help='Multitask architecture', choices=['distral', 'tscl', 'distral_tscl', 'lreg_tscl'], default='cnn')
parser.add_argument('--architecture', help='Policy architecture', choices=['dnc', 'vddnc', 'ddpg', 'vdddpg', 'pathnet', 'pathmatrix', 'dnc'], default='ddpg')
parser.add_argument('--input', help='Input Type', choices=['json', 'feat', 'feason'], default='json')
parser.add_argument('--stage', help='Stage that the model is in', choices=['training', 'evaluation', 'live'], default='evaluation')

class Case():
    def __new__(cls, *args, **kwargs):
        case = super(Case, cls).__new__(cls)
        case._closed = False
        case._spec = None
        return case

    def _work(self): raise NotImplementedError
    def _train(self): raise NotImplementedError
    def _render(self): raise NotImplementedError
    def _evaluate(self): raise NotImplementedError

    model_dir = None
    agent = None

    def close(self):
        pass

    def work(self):
        # USE CPU
        return self._work()

    def train(self):
        # USE GPU
        return self._train()

    def evaluate(self):
        # USE CPU
        return NotImplemented

    def save(self):
        return NotImplemented

    def config(self):
        return NotImplemented

    def render(self):
        return self._render()

class CaseRegistry():
    def __init__(self):
        raise NotImplementedError

    def register(self, entry_point, **kwargs):
        raise NotImplementedError



case_registry = CaseRegistry()

def case(entry_point, **kwargs):
    case_registry.register(entry_point, **kwargs)