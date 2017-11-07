import json
import time
import numpy as np
from omni import omni
from omni.config import MAX_PARAMS

def gen():
    k = 5
    env = omni.instantiate()
    obs = np.zeros(env.observation_space.shape)
    rewards = np.zeros(env.reward_space.shape)
    step = 0

    f = open("./test/test.json", "a+")
    data = []
    for i in range(4):
        random_action = [np.random.rand(1), np.random.rand(1, int(MAX_PARAMS))]
        new_obs, new_rewards, done, info = env.step(random_action)
        step += 1
        step_data = {}

        step_data["obs"] = convert(obs)
        step_data["new_obs"] = convert(new_obs)
        step_data["index_action"] = convert(random_action[0])
        step_data["candidate_action"] = env.spawn(random_action[0], k)
        step_data["param_action"] = convert(random_action[1])
        step_data["rewards"] = convert(rewards)
        step_data["new_rewards"] = convert(new_rewards)
        step_data["done"] = done
        step_data["step"] = step

        data.append(step_data)

        obs = new_obs
        rewards = new_rewards

        time.sleep(300)

    json_request = json.dumps(data)
    f.write(json_request)

def convert(list):
    if isinstance(list, (np.ndarray, np.generic)):
        converted = list.tolist()
        return converted
    else:
        return list

if __name__ == '__main__':
    gen()