import json
import requests
import requests_cache

from omni.interfaces.util import invoke

BASE_URL = 'https://api.coinmarketcap.com/v1/'

coinmarketcap_session = requests_cache.CachedSession(cache_name="coinmarketcap", expire_after=30, backend='sqlite')

def get_all_tickers(input):
    # todo convert integer repr [0.1] into limit
    params = {}
    params['convert'] = input.convert
    #params['limit'] = limit

    return invoke("GET", url=BASE_URL+'ticker/', params=params, session=coinmarketcap_session)

def get_stats(input):

    params = {}
    params['convert'] = input.convert

    return invoke("GET", url=BASE_URL + 'global/', params=params, session=coinmarketcap_session)