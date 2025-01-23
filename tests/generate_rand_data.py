import numpy as np
import gym


def trajbatch2dataset(xs, uss):

    ntrajs, traj_len, state_dim = xs.shape
    _, _, action_dim = uss.shape
    
    # Initialize lists to hold data
    newxs = []  # States
    newxsp = []  # Next states
    newus = []  # Actions

    for i in range(traj_len - 1):  # Iterate over trajectory length - 1
        # Extract states, next states, and actions
        x = xs[:, i, :]  # Current states
        xp = xs[:, i + 1, :]  # Next states
        u = uss[:, i, :]  # Actions

        # Append data
        newxs.append(x)
        newxsp.append(xp)
        newus.append(u)

    # Concatenate all data across trajectories
    states = np.vstack(newxs)  # Shape: (n_samples, state_dim)
    next_states = np.vstack(newxsp)  # Shape: (n_samples, state_dim)
    actions = np.vstack(newus)  # Shape: (n_samples, action_dim)

    return states, actions, next_states


def generate_rand_data(env, ntrajs, traj_len, dt, maxu=100.0, stddev=30.0):

    assert traj_len > 1, "Trajectory length must be greater than 1"
    assert ntrajs >= 1, "Number of trajectories must be at least 1"
    
    # Initialize storage for trajectories
    xs = np.zeros((ntrajs, traj_len, env.observation_space.shape[0]))
    uss = np.zeros((ntrajs, traj_len - 1, 1))
    
    for i in range(ntrajs):
        # Reset environment and initialize trajectory
        if env.spec.id == "Acrobot-v1":
            state=env.reset()
        else:
            env.reset()
            state = env.state
        
        xs[i, 0, :] = state  # Store initial state
        for t in range(traj_len - 1):
            # Generate random action
            action = np.clip(stddev * np.random.randn(), -maxu, maxu)
            action = int(np.clip(action, 0, 1))  # Convert to discrete action
        
            # Step through environment
            next_state, _, done, _ = env.step(action)
            
            # Store states and actions
            xs[i, t + 1, :] = next_state
            uss[i, t, :] = action
            if state[0] < -2.4 or state[0] > 2.4:
                break
    return trajbatch2dataset(xs, uss)


# Example usage
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    ntrajs = 256
    traj_len = 3
    dt = 0.05
    maxu = 100.0
    stddev = 30.0

    states, actions, next_states = generate_rand_data(env, ntrajs, traj_len, dt, maxu=maxu, stddev=stddev)

    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    print("Next States shape:", next_states.shape)