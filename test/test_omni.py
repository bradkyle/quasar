import json
import random
import string
import numpy as np
import sys
from omni.interfaces.processing import process
from omni.config import MAX_LENGTH, CHAR_EMBEDDING

np.set_printoptions(threshold=sys.maxsize)

def test_encode():
    string = "a a"
    correct = np.array(np.concatenate((np.array([0, -1, 0]) ,np.array(9997 * [-1]))))
    response = process(string)
    print(len(response))
    assert len(response) == MAX_LENGTH

def test_big_encode():
    data = [{"type":"exchange","currency":"BTC","amount":"1000","available":"1000","availableForWithdrawal":"1000"},
            {"type":"exchange","currency":"USD","amount":"100000.00","available":"100000.00","availableForWithdrawal":"100000.00"},
            {"type":"exchange","currency":"ETH","amount":"20000","available":"20000","availableForWithdrawal":"20000"}]

    string = json.dumps(data)
    response = process(string)
    print(len(response))
    assert len(response) == MAX_LENGTH

def test_max_encode():
    rand = ''.join([random.choice(list(CHAR_EMBEDDING)) for n in range(MAX_LENGTH+1000)])
    response = process(rand)
    print(len(response))
    assert len(response) == MAX_LENGTH