class Penalty():
    def __init__(self, penalty, store, response):
        print("Not Implemented penalty!")
        
    def dump(self):
        raise NotImplemented


class BadRequestPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class BadConnectionPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class NoneResponsePenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class NotAffordedPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class RateLimitPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class StepPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class NotFoundPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class ResponseSizePenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class RedundancyPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class AffordanceDisabledPenalty(Penalty):
    def __init__(self, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)


class WaitPenalty(Penalty):
    def __init__(self, wait, penalty=75, store='', response=None):
        Penalty.__init__(self, penalty, store, response)