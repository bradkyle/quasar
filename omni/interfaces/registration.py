from omni.error import NoEntryPointError
from gym import error
import pkg_resources
from heapq import nsmallest

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result


class Input():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.args = None

class Interface():
    def __init__(self):
        raise NotImplemented

class InterfaceRegistry():
    def __init__(self):
        raise NotImplemented


# Affordance
# --------------------------------------------------------------------------------------------------------------------->

class Affordance():
    def __init__(self, id, entry_point, enabled=True, sudo_disabled=False, **kwargs):
        self.id = id
        self.sudo_diisabled = sudo_disabled
        self.enabled = enabled

        if entry_point is None:
            raise NoEntryPointError

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

    @property
    def active_affordances(self):
        return self.affordances.items()

    def spawn(self, index, k):
        candidates = nsmallest(k, self.affordances.keys(),
                               key=lambda x: abs(x - (((1 + index) / 2) * len(self.affordances.keys()))))
        return candidates

affordance_registry = AffordanceRegistry()

def affordance(entry_point, **kwargs):
    affordance_registry.register(entry_point, **kwargs)

# Tasks
# --------------------------------------------------------------------------------------------------------------------->

class Task():
    def __init__(self, id, entry_point, enabled=True, sudo_disabled=False, **kwargs):
        self.id = id
        self.sudo_diisabled = sudo_disabled
        self.enabled = enabled

        if entry_point is None:
            raise NoEntryPointError

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


class TaskRegistry():
    def __init__(self):
        self.tasks = {}
        self.id_counter = 0

    def register(self, entry_point, **kwargs):
        self.tasks[self.id_counter] = Task(self.id_counter, entry_point, **kwargs)
        self.id_counter += 1

    def aggregate(self):
        rewards = []
        for task in self.tasks.values():
            rewards.append(float(task())) #todo async
        return rewards

    def list_all(self):
        return self.tasks.items()

task_registry = TaskRegistry()

def task(entry_point, **kwargs):
    task_registry.register(entry_point, **kwargs)

# Closer
# --------------------------------------------------------------------------------------------------------------------->

class Closer():
    def __init__(self, id, entry_point, enabled=True, sudo_disabled=False, **kwargs):
        self.id = id
        self.sudo_diisabled = sudo_disabled
        self.enabled = enabled

        if entry_point is None:
            raise NoEntryPointError

        self.entry_point = entry_point
        self.invoker = load(self.entry_point)
        self.input = Input(**kwargs)

    def __call__(self, *params):
        self.input.args = params
        return self.invoker(self.input)

class CloserRegistry():
    def __init__(self):
        self.closers = {}
        self.id_counter = 0

    def register(self, entry_point, **kwargs):
        self.closers[self.id_counter] = Closer(self.id_counter, entry_point, **kwargs)
        self.id_counter += 1

    def close(self):
        for close in self.closers.values():
            close() #todo async

closer_registry = CloserRegistry()

def closer(entry_point, **kwargs):
    closer_registry.register(entry_point, **kwargs)

# Cache
# --------------------------------------------------------------------------------------------------------------------->

class Cache():
    def __init__(self):
        self.caches = {}

    def __call__(self, *args, **kwargs):
        raise NotImplemented

class CacheRegistry():
    def __init__(self):
        self.caches = {}

    def register(self):
        raise NotImplemented

cache_registry = CacheRegistry()

def cache():
    cache_registry.register()

# Feature
# --------------------------------------------------------------------------------------------------------------------->

class Feature():
    def __init__(self):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        raise NotImplemented


class FeatureRegistry():
    def __init__(self):
        self.features = {}

    def register(self, entry_point, **kwargs):
        raise NotImplemented


feature_registry = FeatureRegistry()

def feature(entry_point, **kwargs):
    feature_registry.register(entry_point, **kwargs)
