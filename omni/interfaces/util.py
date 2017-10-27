import json
import numpy as np
import requests
import time
import re

from omni.config import LINE_LENGTH,PADDING_CHAR,CHAR_EMBEDDING, BAD_REQUEST_PENALTY, SAVE_REQUEST, BAD_CONNECTION_PENALTY

if SAVE_REQUEST:
    f = open("./store.json", "a+")

def invoke(method, url, headers=None, body=None, params=None, payload=None, session=None, encode=True):

    assert method == "GET" or "POST"

    if session is None:
        session = requests.Session()

    req = requests.Request(method, url, params=params, headers=headers, data=body)
    prepared = req.prepare()

    if SAVE_REQUEST:
        r = {}
        r["method"] = prepared.method
        r["url"] = prepared.url
        r["headers"] = {}
        for k, v in prepared.headers.items():
            r["headers"][str(k)] = str(v)
        r["body"] = prepared.body
        r["time"] = time.time()
        if payload:
            r["payload"] = payload
        json_request = json.dumps(r)
        f.write(json_request)

    from omni.interfaces.base.dict import bad_request_penalty, connection_error_penalty

    try:
        response_object = session.send(prepared)
    except requests.exceptions.HTTPError as e:
        bad_request_penalty += BAD_REQUEST_PENALTY
        invoke(method, url, headers, body, params, payload, session, encode)

    except requests.ConnectionError:
        connection_error_penalty += BAD_CONNECTION_PENALTY
        invoke(method, url, headers, body, params, payload, session, encode)

    if encode:
        observation = encode_response(response_object.text)
    else:
        observation = response_object.text
    return observation


def encode_response(observation):
    result = []
    observation = re.sub(' ',  '', observation)
    observation = re.sub('\n', '', observation)
    observation = re.sub('\r', '', observation)
    for line in observation:
        line = list(str(line))
        if len(line) > LINE_LENGTH:
            line = line[-LINE_LENGTH:]
        num_padding = LINE_LENGTH - len(line)
        output_line = line + [PADDING_CHAR] * num_padding
        result.append(np.array([CHAR_EMBEDDING.find(char) for char in output_line], dtype=np.int8))
    print(result[0])
    return result[0]
