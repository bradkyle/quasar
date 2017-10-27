

def penalise_connection_errors(input):
    from omni.interfaces.base.dict import bad_request_penalty
    return_penalty = bad_request_penalty
    bad_request_penalty = 0
    return return_penalty

def penalise_connection_error(input):
    from omni.interfaces.base.dict import connection_error_penalty
    return_penalty = connection_error_penalty
    connection_error_penalty = 0
    return return_penalty

def penalise_response_size(input):
    from omni.interfaces.base.dict import response_size_penalty
    return_penalty = response_size_penalty
    response_size_penalty = 0
    return return_penalty

def step_loss(input):
    return -1