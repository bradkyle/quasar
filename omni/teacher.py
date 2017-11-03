from omni.interfaces.registration import task_registry

class Teacher():
    """
    At each timestep, the Teacher chooses tasks for the Student to practice on.
    The Student trains on those tasks and returns back a score. The Teacher’s
    goal is for the Student to succeed on a final task with as few training steps as possible. Usually the
    task is parameterized by a categorical value representing one of N subtasks, but one can imagine
    also multi-dimensional or continuous task parameterization. The score can be episode total reward in
    reinforcement learning or validation set accuracy in supervised learning.
    """
    def __init__(self, env):
        self.env = env