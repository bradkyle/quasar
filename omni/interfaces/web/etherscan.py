import requests_cache
from omni.interfaces.util import invoke

BASE_URL = "https://etherscan.io/"

etherscan_charts_session = requests_cache.CachedSession(cache_name="etherscan_charts", expire_after=84000, backend='sqlite')

def get_chart(input):
    return invoke("GET", url=BASE_URL + 'chart/' + str(input.chart), session=etherscan_charts_session)