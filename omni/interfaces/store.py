class Store():
    def __init__(self):
        self.store = {}

    def get(self, key):
        if key in self.store:
            return self.store[key]
        else:
            return

    def set(self, key, value):
        self.store[key] = value
        return self.store[key]

    def destroy(self, key):
        if key in self.store:
            del self.store[key]

store = Store()

def get(key):
    return store.get(key)

def set(key, value):
    return store.set(key,value)

def destroy(key):
    return store.destroy(key)