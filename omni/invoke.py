import omni.error
import asyncio
import aiohttp
from omni.interfaces.registration import cache_registry
import time
from omni.interfaces.penalise import BadRequestPenalty, BadConnectionPenalty, NoneResponsePenalty, WaitPenalty
import re
import numpy as np
from omni.config import MAX_LENGTH,PADDING_CHAR,CHAR_EMBEDDING

class Invoker():
    def __init__(self):
        self.cache_registry = cache_registry
        self.q = asyncio.Queue()

    def process(self, input):
        if type(input) is not str:
            input = str(input)

        # replace
        input = re.sub('\n', '', input)
        input = re.sub('\r', '', input)

        # encoding and padding
        input = list(input)
        num_padding = MAX_LENGTH - len(input)
        input = input + [PADDING_CHAR] * num_padding
        result = np.array([CHAR_EMBEDDING.find(char) for char in input], dtype=np.int8)
        return result[:MAX_LENGTH]

    async def invoke(self):

        # check if a cache exists for the request

        # else asynchronously proces the request

        raise None

    async def fetch(self, method, url, headers=None, body=None, params=None, payload=None, session=None, encode=True):
        try:
            response = await aiohttp.request(method=method,url=url,headers=headers,params=params,data=body)

            if await response.text() is None:
                raise omni.error.NoneResponseError("Empty response:" + str(response.status_code))

            if encode:
                observation = self.process(response.text)
            else:
                observation = response.text()

            response.close()
            return observation

        except omni.error.NoneResponseError as e:
            print("Encountered error: " + str(e))
            NoneResponsePenalty()
            await self.fetch(method, url, headers, body, params, payload, session, encode)

        except aiohttp.ClientResponseError as e:
            print("Encountered error: " + str(e))
            BadRequestPenalty()
            await self.fetch(method, url, headers, body, params, payload, session, encode)

        except aiohttp.ClientConnectionError as e:
            print("Encountered error: " + str(e))
            BadConnectionPenalty()
            await self.fetch(method, url, headers, body, params, payload, session, encode)

    async def push(self):
        raise NotImplementedError

invoker = Invoker()

def invoke():
    invoker.invoke()
