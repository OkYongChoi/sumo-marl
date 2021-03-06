# Simulation of Urban Mobility (SUMO) for Multi-Agent Reinforcement Learning (MARL)


This codes has only been confirmed on Windows.
## Install SUMO
. Installing SUMO from [website](https://sumo.dlr.de/docs/Downloads.php) creates SUMO files under "C:\Program Files (x86)\Eclipse\Sumo".

## Install TRaCI package
For SUMO, need to install `TRaCI` for traffic control interface
```shell 
$ pip install traci
```

## Run the code
```python
python main.py
```
R flag, which is intended to mean 'rendering', can show how the agents actually move.

## References
### Multi-Agent papers
 - [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in neural information processing systems 30 (2017).](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

### Traffic Environment
 - Reward definition
   - [Egea, Alvaro Cabrejas, et al. "Assessment of reward functions for reinforcement learning traffic signal control under real-world limitations." 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2020.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9283498)