import re
import numpy as np

from omni.config import MAX_LENGTH,PADDING_CHAR,CHAR_EMBEDDING

def process(input):
    result = []

    # replace
    input = re.sub('\n', '', input)
    input = re.sub('\r', '', input)

    # encoding and padding
    input = list(input)
    num_padding = MAX_LENGTH - len(input)
    input = input + [PADDING_CHAR] * num_padding
    result.append(np.array([CHAR_EMBEDDING.find(char) for char in input], dtype=np.int8))

    return result
