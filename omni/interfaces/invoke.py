import json
import omni.error
import requests
import time
from omni.interfaces.processing import process
from omni.interfaces.penalise import BadRequestPenalty, BadConnectionPenalty, NoneResponsePenalty, WaitPenalty

from omni.config import SAVE_REQUEST

# todo implement with redis
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

    # todo check for none response and return a standard response in that instance

    try:
        response_object = session.send(prepared)
        if encode:
            observation = process(response_object.text)
        else:
            observation = response_object.text

        if observation is None:
            raise omni.error.NoneResponseError

        return observation

    except omni.error.NoneResponseError:
        NoneResponsePenalty()
        WaitPenalty(2)
        time.sleep(2)
        invoke(method, url, headers, body, params, payload, session, encode)

    except requests.exceptions.HTTPError:
        BadRequestPenalty()
        WaitPenalty(2)
        time.sleep(2)
        invoke(method, url, headers, body, params, payload, session, encode)

    except requests.ConnectionError:
        BadConnectionPenalty()
        WaitPenalty(2)
        time.sleep(2)
        invoke(method, url, headers, body, params, payload, session, encode)
