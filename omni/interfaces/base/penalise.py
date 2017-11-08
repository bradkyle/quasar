from omni.interfaces.store import get, set

def aggregate_penalty(penalty_name):
    counter = get(penalty_name)
    if counter is not None:
        set(penalty_name, 0)
        return counter
    else:
        counter = set(penalty_name, 0)
        return counter


