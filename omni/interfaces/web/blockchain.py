import requests_cache

from omni.interfaces.invoke import invoke

BASE_URI = "https://api.blockchain.info"

charts_session = requests_cache.CachedSession(cache_name="blockchain_charts", expire_after=84000, backend='sqlite')

# Affordances
# =====================================================================================================================>

def get_chart(input):

    # chart, timespan=None, rollingAverage=None, start=None, sampled=None

    params = {}

    # params['rollingAverage'] = max_rolling_average * input.args[0]
    # params['start'] = max_start * input.args[1]
    # params['timespan'] = start -> max_timespan * input.args[2]
    # params['include_breaks'] = input.args[3] > 0.5 : True else False

    url = "https://api.blockchain.info/charts/" + input.chart
    return invoke("GET", url, params=params, session=charts_session)


info_session = requests_cache.CachedSession(cache_name="blockchain_info", expire_after=900, backend='sqlite')
def get_ticker(input):
    url = BASE_URI+"/ticker"
    return invoke("GET", url, session=info_session)

def get_stats(input):
    url = BASE_URI+"/stats"
    return invoke("GET", url, session=info_session)

def get_pools(input):
    url = BASE_URI+"/pools"
    return invoke("GET", url, session=info_session)

# Features
# =====================================================================================================================>

def get_todays_chart(input):

    # chart, timespan=None, rollingAverage=None, start=None, sampled=None

    params = {}

    # params['rollingAverage'] = max_rolling_average * input.args[0]
    # params['start'] = max_start * input.args[1]
    params['timespan'] = '1days'
    # params['include_breaks'] = input.args[3] > 0.5 : True else False

    url = "https://api.blockchain.info/charts/" + input.chart
    return invoke("GET", url, params=params, session=charts_session, encode=False)