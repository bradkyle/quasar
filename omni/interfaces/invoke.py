import json
import omni.error
import requests
import time
from omni.interfaces.processing import process
from omni.interfaces.penalise import BadRequestPenalty, BadConnectionPenalty, NoneResponsePenalty, WaitPenalty

def invoke(method, url, headers=None, body=None, params=None, payload=None, session=None, encode=True):

    assert method == "GET" or "POST"

    if session is None:
        session = requests.Session()

    req = requests.Request(method, url, params=params, headers=headers, data=body)
    prepared = req.prepare()

    try:
        response_object = session.send(prepared)
        if response_object.text is None:
            raise omni.error.NoneResponseError("There was no content in the response, status code:" +str(response_object.status_code))

        if encode:
            observation = process(response_object.text)
        else:
            observation = response_object.text

        return observation

    except requests.ConnectionError as e:
        print("Encountered error: " + str(e))
        BadConnectionPenalty()
        WaitPenalty(10)
        time.sleep(10)
        invoke(method, url, headers, body, params, payload, session, encode)

    except requests.exceptions.HTTPError as e:
        print("Encountered error: " + str(e))
        BadRequestPenalty()
        WaitPenalty(10)
        time.sleep(10)
        invoke(method, url, headers, body, params, payload, session, encode)

    except omni.error.NoneResponseError as e:
        print("Encountered error: "+ str(e))
        NoneResponsePenalty()
        WaitPenalty(10)
        time.sleep(10)
        invoke(method, url, headers, body, params, payload, session, encode)



