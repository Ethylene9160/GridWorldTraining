
import sys
from pathlib import Path

# Ensure project root (one level up) is on sys.path so `from src.grid_world import GridWorld` works
# This is more robust than a plain `sys.path.append("..")` because it uses an absolute path.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":           
    env = GridWorld()
    state = env.reset()               
    for t in range(1):
        env.render()
        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break
    
    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    print("Policy Matrix: ", policy_matrix.shape)
    env.add_policy(policy_matrix)

    
    # Add state values
    values = np.random.uniform(0,10,(env.num_states,))
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2)