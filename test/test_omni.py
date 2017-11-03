import numpy as np
import sys
from omni.interfaces.processing import process

np.set_printoptions(threshold=sys.maxsize)

def test_encode():
    string = "a a"
    correct = np.array(np.concatenate((np.array([0, -1, 0]) ,np.array(9997 * [-1]))))
    response = process(string)
    print(correct)
    assert response == correct
