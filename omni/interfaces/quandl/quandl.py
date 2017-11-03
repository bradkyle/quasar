from omni.interfaces.invoke import invoke

def search(input):


    """
    For a list of all databases on Quandl, do this:

    https://www.quandl.com/api/v3/databases
    For a list of datasets in a given database, do this:

    https://www.quandl.com/api/v3/databases/WIKI/codes.json
    """
    params = {}
    #column_index = 4 & exclude_column_names = true & rows = 3 & start_date = 2012 - 11 - 01 & end_date = 2013 - 11 - 30 & order = asc & collapse = quarterly & transform = rdiff

    return invoke("GET", url="https://www.quandl.com/api/v3/datasets/" + str(input.term)+"/data.json", params=params)