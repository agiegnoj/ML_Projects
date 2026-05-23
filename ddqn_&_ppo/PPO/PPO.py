import gymnasium as gym
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Network import Network

def training(epochs=600, numUpdates=5, epsilon = 0.2, batchSize=1024, subBatchSize=64):
    """PPO algorithm, quickly adaptle for different learning tasks"""
    env = gym.make()
    numActions = env.action_space.shape[0]
    obs, _ = env.reset()

    numObservations = obs.size
    actor = Network(numObservations, numActions)
    critic = Network(numObservations, 1)
    logStd = torch.nn.Parameter(torch.zeros(numActions))
    optimizerActor = optim.Adam(list(actor.parameters()) + [logStd], lr=1e-4)
    optimizerCritic = optim.Adam(critic.parameters(), lr=1e-4)


    for epoch in tqdm(range(epochs)):
         (batchStates, batchActions,
          batchLogProbs, batchRewardsToGo) = getRollout(actor, env, logStd, timeStepsPerBatch=batchSize)
         dataSet = TensorDataset(batchStates, batchActions, batchLogProbs, batchRewardsToGo)
         dl = DataLoader(dataSet, batch_size=subBatchSize, shuffle=True)


         for bS, bA, bLP, bRTG in dl:
             for i in range(numUpdates):
                 with torch.no_grad():
                   V = critic(bS).squeeze()

                 advantages = bRTG - V
                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                 V, currentLogProbs = evaluate(actor, critic, bS, bA, logStd)

                 ratios = torch.exp(currentLogProbs - bLP)
                 ratios = torch.clamp(ratios, 0, 10)
                 surrogate1 = ratios * advantages
                 surrogate2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
                 actorLoss = torch.mean(-torch.min(surrogate1, surrogate2))
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
    std = torch.exp(logStd)
    std = torch.clamp(std, 1e-3, 1.0)
    covarianceMatrix = torch.diag_embed(std**2)
    V = critic(batchObservations).squeeze()
    mean = actor(batchObservations)
    mean = torch.clamp(mean, -10, 10)
    dist = torch.distributions.MultivariateNormal(mean, covarianceMatrix)
    logProbs = dist.log_prob(batchActions)
    return V, logProbs


def getAction(actor, observation, logStd):
    std = torch.exp(logStd)
    std = torch.clamp(std, 1e-3, 1.0)
    covarianceMatrix = torch.diag_embed(std**2)
    mean = actor(observation)
    mean = torch.clamp(mean, -10, 10)
    dist = torch.distributions.MultivariateNormal(mean, covarianceMatrix)
    action = dist.sample()
    logProb = dist.log_prob(action)
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
        t2 = 0
        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            batchStates.append(state)
            t +=1
            t2 +=1

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
        r= 0
        episodeRewardsToGo = []
        for reward in reversed(episodeRewards):
            r = reward + gamma *r
            episodeRewardsToGo.append(r)
        episodeRewardsToGo.reverse()

        batchRewardsToGo.extend(episodeRewardsToGo)

    batchStates = torch.cat(batchStates)
    batchActions = torch.tensor(batchActions, dtype=torch.float32)
    batchLogProbs = torch.tensor(batchLogProbs, dtype=torch.float32)
    batchRewardsToGo = torch.tensor(batchRewardsToGo, dtype=torch.float32)


    return batchStates, batchActions, batchLogProbs, batchRewardsToGo


if __name__ == "__main__":
   agent = training()
   torch.save(agent.state_dict(), f"PPOAgent_{"baseModel"}_{"v1"}.pt")
