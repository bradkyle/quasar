from omni.interfaces.registration import affordance_registry

class Randomiser():
    """
    Afoordance Level Domain Randomization for Transferring
    Deep Neural Networks from Simulation to the Real World.
    Switch from training to reality whereby.
    Give each interface a domain i.e. dev,
    g1, g2, g3
    """
    def __init__(self, env, entropy_rate):
        self.env = env