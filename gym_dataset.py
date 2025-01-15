#code from tenserflow 
import gym

#set env
env=gym.make('CartPole-v1')
for e in range(20): #epi
    obs=env.reset()
    for t in range(100):
        env.render()
        print(obs) #package pygame 통해서 cartpole env 관찰
        action=env.action_space.sample() #좌,우 값이 0,1로 랜덤하게 전달
        obs, reward, done, info = env.step(action)
        if done:
            print('finished')
            break
env.close()
