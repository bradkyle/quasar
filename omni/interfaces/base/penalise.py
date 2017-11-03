

def penalise_connection_errors(input):
    from omni.interfaces.store import bad_request_penalty
    return_penalty = bad_request_penalty
    bad_request_penalty = 0
    return return_penalty

def penalise_connection_error(input):
    from omni.interfaces.store import bad_connection_penalty
    return_penalty = bad_connection_penalty
    bad_connection_penalty = 0
    return return_penalty

def penalise_response_size(input):
    from omni.interfaces.store import response_size_penalty
    return_penalty = response_size_penalty
    response_size_penalty = 0
    return return_penalty

def step_loss(input):
    return -1