import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Network import Network

def training(epochs=600, numUpdates=5, epsilon = 0.2, batchSize=1024, subBatchSize=64):
    """PPO algorithm, quickly adaptable for different learning tasks"""
    env = gym.make()
    numActions = env.action_space.shape[0]
    obs, _ = env.reset()

    numObservations = obs.size
    actor = Network(numObservations, numActions)
    critic = Network(numObservations, 1)
    logStd = torch.nn.Parameter(torch.zeros(numActions))
    entropyCoeefficient = torch.nn.Parameter(torch.tensor([0.001], dtype=torch.float32))
    optimizerActor = optim.Adam(list(actor.parameters()) + [logStd] + [entropyCoeefficient], lr=3e-4)
    optimizerCritic = optim.Adam(critic.parameters(), lr=3e-4)

    for epoch in tqdm(range(epochs)):
        (batchStates, batchActions,
         batchLogProbs, batchRewardsToGo) = getRollout(actor, env, logStd, timeStepsPerBatch=batchSize)
        dataSet = TensorDataset(batchStates, batchActions, batchLogProbs, batchRewardsToGo)
        dl = DataLoader(dataSet, batch_size=subBatchSize, shuffle=True)

        for bS, bA, bLP, bRTG in dl:
            with torch.no_grad():
                V = critic(bS).squeeze()
            advantages = bRTG - V
            advantages = torch.clamp(advantages, -50, 50)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            for i in range(numUpdates):
                V, currentLogProbs, entropy = evaluate(actor, critic, bS, bA, logStd)

                logRatio = currentLogProbs - bLP
                logRatio = torch.clamp(logRatio, -20, 20)
                ratios = torch.exp(logRatio)

                surrogate1 = ratios * advantages
                surrogate2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
                entropyLoss = entropy.mean()
                actorLoss = torch.mean(-torch.min(surrogate1, surrogate2)) - entropyCoeefficient * entropyLoss
                optimizerActor.zero_grad()
                actorLoss.backward()

                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizerActor.step()

                criterion = nn.MSELoss()
                criticLoss = criterion(V, bRTG)
                optimizerCritic.zero_grad()
                criticLoss.backward()

                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizerCritic.step()

    return actor


def evaluate(actor, critic, batchObservations, batchActions, logStd):
    logStd = torch.clamp(logStd, -1, 2)
    std = torch.exp(logStd)
    mean = actor(batchObservations)
    mean = torch.clamp(mean, -5, 5)

    dist = torch.distributions.Normal(mean, std)
    dist = torch.distributions.Independent(dist, 1)
    logProbs = dist.log_prob(batchActions).sum(dim=-1)

    V = critic(batchObservations).squeeze()

    return V, logProbs, dist.entropy()


def getAction(actor, observation, logStd):
    logStd = torch.clamp(logStd, -1, 2)
    std = torch.exp(logStd)
    mean = actor(observation)
    mean = torch.clamp(mean, -5, 5)
    dist = torch.distributions.Normal(mean, std)
    dist = torch.distributions.Independent(dist, 1)

    action = dist.sample()
    action = torch.clamp(action, -1, 1)
    logProb = dist.log_prob(action).sum(dim=-1)

    return action.detach(), logProb.detach()


def getRollout(actor, env, logStd, timeStepsPerBatch=1024, gamma=0.95):
    batchStates = []
    batchActions = []
    batchRewards = []
    batchLogProbs = []
    batchRewardsToGo = []

    t = 0
    while t < timeStepsPerBatch:
        state, _ = env.reset()
        episodeRewards = []

        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            batchStates.append(state)
            t += 1

            action, logProb = getAction(actor, state, logStd)
            action = action.squeeze(0).numpy()

            state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            episodeRewards.append(reward)
            batchActions.append(action)
            batchLogProbs.append(logProb)

            done = terminated or truncated

        batchRewards.append(episodeRewards)

    for episodeRewards in batchRewards:
        r = 0
        episodeRewardsToGo = []
        for reward in reversed(episodeRewards):
            r = reward + gamma * r
            episodeRewardsToGo.append(r)
        episodeRewardsToGo.reverse()

        batchRewardsToGo.extend(episodeRewardsToGo)

    batchStates = torch.tensor(np.array(batchStates), dtype=torch.float32)
    batchActions = torch.tensor(batchActions, dtype=torch.float32)
    batchLogProbs = torch.tensor(np.array(batchLogProbs), dtype=torch.float32)
    batchRewardsToGo = torch.tensor(batchRewardsToGo, dtype=torch.float32)

    return batchStates, batchActions, batchLogProbs, batchRewardsToGo


if __name__ == "__main__":
   agent = training()
   torch.save(agent.state_dict(), f"PPOAgent_{"baseModel"}_{"v1"}.pt")
