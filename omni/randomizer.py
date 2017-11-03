from omni.interfaces.registration import affordance_registry

class Randomiser():
    def __init__(self, env, entropy_rate):
        self.env = env