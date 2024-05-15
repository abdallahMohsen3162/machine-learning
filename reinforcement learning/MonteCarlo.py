import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
history = []
def run(is_training=True, render=False):
 
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
 
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095,.2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)
 
    def choose_action(state_p, state_v, state_a, state_av, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(q[state_p, state_v, state_a, state_av, :])
 
    if(is_training):
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()
 
    learning_rate_a = 0.1 
    discount_factor_g = 0.99 
    episodes = 100000
    epsilon = 1      
    epsilon_decay_rate = 0.00001 
    rng = np.random.default_rng() 
 
    rewards_per_episode = []
 
    i = 0
 
    for episode in range(episodes):
        
        state = env.reset()[0]    
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)
 
        action = choose_action(state_p, state_v, state_a, state_av, epsilon)
 
        terminated = False          # True when reached goal
 
        total_reward=0
        episode_states = []
        episode_actions = []
        episode_rewards = []
 
        while(not terminated and total_reward < 10000):
 
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)
 
            episode_states.append((state_p, state_v, state_a, state_av))
            episode_actions.append(action)
            episode_rewards.append(reward)
 
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av= new_state_av
            action = choose_action(state_p, state_v, state_a, state_av, epsilon)
 
            total_reward+=reward
 
            if render:
                env.render()
        
        rewards_per_episode.append(total_reward)

        if is_training:
            for t in range(len(episode_states)):
                state_p, state_v, state_a, state_av = episode_states[t]
                action = episode_actions[t]
                reward = episode_rewards[t]
                G = 0
                for k in range(t, len(episode_rewards)):
                    G += discount_factor_g**(k-t) * episode_rewards[k]
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (G - q[state_p, state_v, state_a, state_av, action])
 

        epsilon = max(epsilon - epsilon_decay_rate, 0.1)
        if render:
            print(rewards_per_episode)

        if (episode + 1) % 100 == 0:
            mean_rewards = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Mean Rewards (last 100): {mean_rewards:.2f}")
            history.append(mean_rewards)
            rewards_per_episode = []
 
            # Check if environment is solved
            if mean_rewards >= 195.0:
                print(f"Environment solved in {episode + 1} episodes!")
                break
 
    env.close()
 

    if is_training:
        f = open('cartpole.pkl','wb')
        pickle.dump(q, f)
        f.close()
    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
        
    plt.plot(history)
    plt.savefig(f'cartpoleMonteCarlo.png')
    plt.show()
    
if __name__ == '__main__':
    run(is_training=True, render=True)
 
