import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Network import Network

def training(epochs=400, numUpdates=10, epsilon = 0.2, batchSize=2048, subBatchSize=256):
    """PPO algorithm, quickly adaptable for different learning tasks"""
    env = gym.make()
    numActions = env.action_space.shape[0]
    obs, _ = env.reset()

    numObservations = obs.size
    actor = Network(numObservations, numActions)
    critic = Network(numObservations, 1)
    logStd = torch.nn.Parameter(torch.zeros(numActions))
    entropyCoeefficient = 0.001
    optimizerActor = optim.Adam(list(actor.parameters()) + [logStd], lr=3e-4)
    optimizerCritic = optim.Adam(critic.parameters(), lr=1e-4)

    for epoch in tqdm(range(epochs)):
        (batchStates, batchActions,
         batchLogProbs, batchRewardsToGo) = getRollout(actor, env, logStd, timeStepsPerBatch=batchSize)
        dataSet = TensorDataset(batchStates, batchActions, batchLogProbs, batchRewardsToGo)
        dl = DataLoader(dataSet, batch_size=subBatchSize, shuffle=True)

        for bS, bA, bLP, bRTG in dl:
            with torch.no_grad():
                V = critic(bS).squeeze()
            advantages = bRTG - V
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            for i in range(numUpdates):
                V, currentLogProbs, entropy = evaluate(actor, critic, bS, bA, logStd)

                logRatio = currentLogProbs - bLP
                logRatio = torch.clamp(logRatio, -5, 5)
                ratios = torch.exp(logRatio)

                surrogate1 = ratios * advantages
                surrogate2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
                entropyLoss = entropy.mean()
                actorLoss = torch.mean(-torch.min(surrogate1, surrogate2)) - entropyCoeefficient * entropyLoss
                optimizerActor.zero_grad()
                actorLoss.backward()

                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optimizerActor.step()

                criterion = nn.MSELoss()
                criticLoss = criterion(V, bRTG)
                optimizerCritic.zero_grad()
                criticLoss.backward()

                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizerCritic.step()

    return actor


def evaluate(actor, critic, batchObservations, batchActions, logStd):
    logStd = torch.clamp(logStd, -2, 0.5)
    std = torch.exp(logStd).clamp(min=1e-6, max=10)
    mean = actor(batchObservations)
    mean = torch.clamp(mean, -5, 5)

    dist = torch.distributions.Normal(mean, std)
    dist = torch.distributions.Independent(dist, 1)
    logProbs = dist.log_prob(batchActions)

    V = critic(batchObservations).squeeze()

    return V, logProbs, dist.entropy()


def getAction(actor, observation, logStd):
    logStd = torch.clamp(logStd, -2, 0.5)
    std = torch.exp(logStd).clamp(min=1e-6, max=10)
    mean = actor(observation)
    mean = torch.clamp(mean, -5, 5)
    dist = torch.distributions.Normal(mean, std)
    dist = torch.distributions.Independent(dist, 1)

    action = dist.sample()
    logProb = dist.log_prob(action)

    return action.detach(), logProb.item()


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

    batchStates = torch.cat(batchStates, dim=0)
    batchActions = torch.tensor(batchActions, dtype=torch.float32)
    batchLogProbs = torch.tensor(np.array(batchLogProbs), dtype=torch.float32)
    batchRewardsToGo = torch.tensor(batchRewardsToGo, dtype=torch.float32)

    return batchStates, batchActions, batchLogProbs, batchRewardsToGo


if __name__ == "__main__":
   agent = training()
   torch.save(agent.state_dict(), f"PPOAgent_{"baseModel"}_{"v1"}.pt")
