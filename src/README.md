# MAG
This code accompanies the paper "Models as Agents: Optimizing Multi-Step Predictions of Interactive Local Models
in Model-Based Multi-Agent Reinforcement Learning".

The repository contains MAG implementation as well as fine-tuned hyperparameters in ```configs/dreamer/optimal``` folder.

## Usage

```
python3 train.py --n_workers 2 --starcraft
```


## SMAC

<img height="300" alt="starcraft" src="https://user-images.githubusercontent.com/22059171/152656435-1634c15b-ca6d-4b23-9383-72fe3759b9e3.png">

The code for the environment can be found at 
[https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)


## Code Structure

- ```agent``` contains implementation of MAMBA 
  - ```controllers``` contains logic for inference
  - ```learners``` contains logic for learning the agent
  - ```memory``` contains buffer implementation
  - ```models``` contains architecture of MAMBA
  - ```optim``` contains logic for optimizing loss functions
  - ```runners``` contains logic for running multiple workers
  - ```utils``` contains helper functions
  - ```workers``` contains logic for interacting with environment
- ```env``` contains environment logic
- ```networks``` contains neural network architectures
