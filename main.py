import gym
import time
from agent import Agent
import numpy as np

n_games = 10000000

agent = Agent()
env = gym.make('BipedalWalker-v2')

scores = []
iter = 0
for i_episode in range(n_games):
    state = env.reset()

    score = 0

    while True:
        # env.render()

        action = agent(state)
        # action is vector of size 4
        # observation is vector of size 24
        newState, reward, done, info = env.step(action)
        agent.addToBatch(state, action)
        iter += 1

        if(iter % 100 == 0):
            agent.train(np.mean(scores) if scores != [] else 0)

        score += reward
        state = newState
        if done:
            break
    agent.endGame(score)
    scores.append(score)
    if(i_episode % 500 == 0):
        print("Episode "+str(i_episode)+" finished.")

    if(len(scores) >= 10):
        scores = scores[1:]
