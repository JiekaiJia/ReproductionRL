from typing import List


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    ret = []
    rewards.reverse()
    for r in rewards:
        if len(ret) == 0:
            ret.append(r)
        else:
            reward_to_go = ret[-1] + gamma * r
            ret.append(reward_to_go)
    ret.reverse()
    return ret
