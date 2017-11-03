from omni.interfaces.registration import affordance_registry, task_registry
from omni.interfaces.processing import process

def list_affordances():
    all_affordances = affordance_registry.list_all()
    return process(all_affordances)

def list_tasks():
    all_tasks = task_registry.list_all()
    return process(all_tasks)