import json

import numpy as np
import six

from omni import Omni

class Envs(object):
    def __init__(self):
        self.omni = Omni()

    def _lookup_instance(self, instance_id):
        try:
            return self.omni.instances[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def _remove_instance(self, instance_id):
        try:
            del self.omni.instances[instance_id]
        except:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def create(self, seed=None):
        instance = self.omni.instantiate()
        if seed:
            instance.seed(seed)
        return instance.instance_id

    def reset(self, instance_id):
        instance = self._lookup_instance(instance_id)
        instance.reset()
        return None

    def step(self, instance_id, action):
        instance = self._lookup_instance(instance_id)
        if isinstance(action, six.integer_types):
            nice_action = action
        else:
            nice_action = np.array(action)
        [observation, reward, done, info] = instance.step(nice_action)
        obs_jsonable = instance.observation_space.to_jsonable(observation)
        return [obs_jsonable, reward, done, info]

    def get_action_space_contains(self, instance_id, x):
        instance = self._lookup_instance(instance_id)
        return instance.action_space.contains(int(x))

    def get_action_space_info(self, instance_id):
        instance = self._lookup_instance(instance_id)
        return self._get_space_properties(instance.action_space)

    def get_action_space_sample(self, instance_id):
        instance = self._lookup_instance(instance_id)
        action = instance.action_space.sample()
        if isinstance(action, (list, tuple)) or ('numpy' in str(type(action))):
            try:
                action = action.tolist()
            except TypeError:
                print(type(action))
                print('TypeError')
        return action

    def get_observation_space_contains(self, instance_id, j):
        instance = self._lookup_instance(instance_id)
        info = self._get_space_properties(instance.observation_space)
        for key, value in j.items():
            # Convert both values to json for comparibility
            if json.dumps(info[key]) != json.dumps(value):
                print('Values for "{}" do not match. Passed "{}", Observed "{}".'.format(key, value, info[key]))
                return False
        return True

    def get_observation_space_info(self, instance_id):
        instance = self._lookup_instance(instance_id)
        return self._get_space_properties(instance.observation_space)

    def _get_space_properties(self, space):
        info = {}
        info['name'] = space.__class__.__name__
        if info['name'] == 'Discrete':
            info['n'] = space.n
        elif info['name'] == 'Box':
            info['shape'] = space.shape
            # It's not JSON compliant to have Infinity, -Infinity, NaN.
            # Many newer JSON parsers allow it, but many don't. Notably python json
            # module can read and write such floats. So we only here fix "export version",
            # also make it flat.
            info['low']  = [(x if x != -np.inf else -1e100) for x in np.array(space.low ).flatten()]
            info['high'] = [(x if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]
        elif info['name'] == 'HighLow':
            info['num_rows'] = space.num_rows
            info['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in np.array(space.matrix).flatten()]
        return info

    def instance_close(self, instance_id):
        instance = self._lookup_instance(instance_id)
        instance.close()
        self._remove_instance(instance_id)

class InvalidUsage(Exception):
    status_code = 400
    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv
