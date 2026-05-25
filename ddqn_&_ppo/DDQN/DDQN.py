import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
from MemoryBuffer import *
from QNetwork import *


def training(batchSize=64, gamma=0.99, epsilon=1.0, tau=0.005, lr=3e-4, episodes=1000):
    """DQN algorithm, quickly adaptable for different learning tasks"""

    env = gym.make()

    nActions = env.action_space.n

    state, info = env.reset()
    nObservations = state.size

    memory = MemoryBuffer(240000)

    policyNetwork = QNetwork(nObservations, nActions)
    targetNetwork =QNetwork(nObservations, nActions)

    optimizer = optim.Adam( policyNetwork.parameters(),lr=lr,amsgrad=True)

    targetNetwork.load_state_dict(policyNetwork.state_dict())
    targetNetwork.eval()
    policyNetwork.train()

    progress = tqdm(range(episodes))
    decayStep = epsilon/(0.5*episodes)


    for e in progress:
        state, info = env.reset()
        done = False

        while not done:

            stateTensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

            if random.random() > epsilon:
                with torch.no_grad():
                    action = (policyNetwork(stateTensor).argmax(dim=1).view(1, 1))
            else:
                action = torch.tensor([[env.action_space.sample()]], dtype=torch.long)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            env.render()

            done = terminated or truncated

            reward = torch.tensor([reward], dtype=torch.float32)

            nextState = observation if not done else None
            nextStateTensor = torch.tensor(nextState, dtype=torch.float32).flatten().unsqueeze(0) if nextState is not None else None

            memory.add(stateTensor, action, nextStateTensor, reward)
            state = nextState

            if len(memory) > batchSize*4:
               optimizeModel(policyNetwork,targetNetwork,optimizer,memory,batchSize,gamma)

        epsilon = max(0.05, epsilon - decayStep)

        with torch.no_grad():
            for target_p, policy_p in zip(targetNetwork.parameters(), policyNetwork.parameters()):
                target_p.data.copy_(tau * policy_p.data +(1.0 - tau) * target_p.data)

    return policyNetwork


def optimizeModel(policyNetwork,targetNetwork, optimizer,memory,batchSize,gamma):
    if len(memory) < batchSize:
        return

    transitions = memory.sample(batchSize)
    batch = Transition(*zip(*transitions))

    nonFinalMask = torch.tensor(tuple(s is not None for s in batch.next_state),dtype=torch.bool)

    nonFinalNextStates = torch.cat([s for s in batch.next_state if s is not None])

    stateBatch = torch.cat(batch.state)
    actionBatch = torch.cat(batch.action)
    rewardBatch = torch.cat(batch.reward)

    qValues = policyNetwork(stateBatch)

    stateActionValues = qValues.gather(1, actionBatch).squeeze(1)

    nextStateValues = torch.zeros(batchSize)

    with torch.no_grad():
        nextActions = policyNetwork(nonFinalNextStates).argmax(1, keepdim=True)
        nextStateValues[nonFinalMask] = (
            targetNetwork(nonFinalNextStates)
            .gather(1, nextActions)
            .squeeze(1)
        )

    expectedValues = rewardBatch + gamma * nextStateValues

    criterion = nn.SmoothL1Loss()
    loss = criterion(stateActionValues, expectedValues)

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(policyNetwork.parameters(), 1.0)
    optimizer.step()



if __name__ == "__main__":
   agent = training()
   torch.save(agent.state_dict(), f"DQNAgent_{"baseModel"}_{"v1"}.pt")
