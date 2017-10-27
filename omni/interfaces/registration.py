from functools import partial
import itertools
from gym import error
import pkg_resources

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result


class Input():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.args = None


# Affordance
# --------------------------------------------------------------------------------------------------------------------->

class Affordance():
    def __init__(self, id, entry_point=None, **kwargs):
        self.id = id

        if entry_point is None:
            raise Exception

        self.entry_point = entry_point
        self.invoker = load(self.entry_point)
        self.input = Input(**kwargs)

    def __call__(self, *params):
        self.input.args = params
        return self.invoker(self.input)

    @property
    def _info(self):
        return {
            "index": self.id,
            "entry_point": self.entry_point
        }

class AffordanceRegistry():
    def __init__(self):
        self.affordances = {}
        self.id_counter = 0

    def register(self, entry_point, **kwargs):
        self.affordances[self.id_counter] = Affordance(self.id_counter, entry_point, **kwargs)
        self.id_counter += 1

    def lookup(self, id):
        return self._find(id)

    def _find(self, id):
        if not id in self.affordances:
            raise error.Error('Could not find affordance with id: {}'.format(id))
        else:
            return self.affordances[id]

    def list_all(self):
        return self.affordances.items()

affordance_registry = AffordanceRegistry()

def register(entry_point, **kwargs):
    affordance_registry.register(entry_point, **kwargs)

# Tasks
# --------------------------------------------------------------------------------------------------------------------->

class Task():
    def __init__(self, id, entry_point=None, **kwargs):
        self.id = id
        self.entry_point = entry_point

        if entry_point is None:
            raise Exception

        self.invoker = load(self.entry_point)
        self.input = Input(**kwargs)

    def __call__(self, *params):
        self.input.args = params
        return self.invoker(self.input)

    @property
    def _info(self):
        return {
            "index": self.id,
            "entry_point": self.entry_point
        }


class TaskRegistry():
    def __init__(self):
        self.tasks = {}
        self.id_counter = 0

    def task(self, entry_point, **kwargs):
        self.tasks[self.id_counter] = Task(self.id_counter, entry_point, **kwargs)
        self.id_counter += 1

    def aggregate(self):
        rewards = []
        for task in self.tasks.values():
            rewards.append(float(task()))
        return rewards

    def list_all(self):
        return self.tasks.items()

task_registry = TaskRegistry()

def task(entry_point, **kwargs):
    task_registry.task(entry_point, **kwargs)

# Closer
# --------------------------------------------------------------------------------------------------------------------->

class Closer():
    def __init__(self, id, entry_point=None, cached=False, cache_length = None, **kwargs):
        self.id = id
        self.entry_point = entry_point
        self.cached = cached
        self.cache_length = cache_length

        self.invoker = load(self.entry_point)
        self.input = Input(**kwargs)

    def __call__(self, *params):
        self.input.args = params
        return self.invoker(self.input)

class CloserRegistry():
    def __init__(self):
        self.closers = {}
        self.id_counter = 0

    def closer(self, entry_point, **kwargs):
        self.closers[self.id_counter] = Closer(self.id_counter, entry_point, **kwargs)
        self.id_counter += 1

    def close(self):
        for close in self.closers.values():
            close()

closer_registry = CloserRegistry()

def closer(entry_point, **kwargs):
    closer_registry.closer(entry_point, **kwargs)