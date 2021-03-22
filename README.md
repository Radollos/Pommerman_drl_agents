# Pommerman_drl_agents
Deep reinfrocement learning agents for Pommerman game from https://www.pommerman.com/.

Includes agents based on Deep Q-Network, Proximal Policy Optimization and Advantage Actor Critic. 

The table below presents achived results for PPO agent with diffrent model architecture. Models were tested with 1000 games.

| Number of hidden layers | Number of neurons | Winning game number | Average game time | 
|---|---|---|---|
| 1 | 500 | 251 | 307.192 |
| 1 | 1000 | 286 | 328.534 |
| 2 | 250125 | 330 | 321.073 |
| 2 | 500250 | 447 | 298.983 |
| 2 | 700350 | **456** | 292.461 |

The based model (simple heuristic proposed by authours Pommperman competition) achived results 220 winning games per 1000.
