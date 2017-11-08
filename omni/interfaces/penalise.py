from omni.interfaces.store import get, set

# todo fortify inheritance

class Penalty():
    def __init__(self, penalty, penalty_name, response):
        counter = get(penalty_name)
        if counter:
            counter += penalty
        else:
            set(penalty_name, penalty)

class BadRequestPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='bad_request_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class BadConnectionPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='bad_connection_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class NoneResponsePenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='none_response_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class NotAffordedPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='not_afforded_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class RateLimitPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='rate_limit_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class StepPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='step_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class NotFoundPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='not_found_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class ResponseSizePenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='response_size_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class AffordanceDisabledPenalty(Penalty):
    def __init__(self, penalty=75, penalty_name='affordance_disabled_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)


class WaitPenalty(Penalty):
    def __init__(self, penalty=-75, penalty_name='wait_penalty', response=None):
        Penalty.__init__(self, penalty, penalty_name, response)