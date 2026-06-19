# Pacman Q-Learning Agent

This repository contains my Q-learning implementation for a Pacman-based reinforcement learning coursework task in the **Machine Learning** module at King’s College London.

The coursework required implementing the Q-learning algorithm inside `mlLearningAgents.py` so that Pacman could learn to choose better actions through repeated gameplay.

---

## Project Timeline

* **Originally completed:** 2025
* **Published on GitHub:** 2026
* **Context:** MSc Artificial Intelligence Machine Learning coursework project

This repository has been cleaned and documented for portfolio purposes.

---

## Project Overview

The goal of this project was to train a Pacman agent using **Q-learning**.

The agent learns by playing multiple episodes of Pacman. During training, it updates Q-values based on the current state, selected action, reward, next state, and future expected rewards. After training, exploration and learning are disabled so the agent follows the learned policy.

This repository contains only my implemented `mlLearningAgents.py` file, not the full Pacman framework.

---

## Reinforcement Learning Method

The implementation is based on **Q-learning**, a model-free reinforcement learning algorithm.

The agent uses:

* Q-value storage
* epsilon-greedy exploration
* learning rate `alpha`
* discount factor `gamma`
* reward-based updates
* training and testing episode separation
* policy selection from learned Q-values

---

## Technologies Used

* Python
* Reinforcement Learning
* Q-learning
* Pacman AI environment
* Dictionary-based Q-value representation
* Epsilon-greedy action selection

---

## Repository Structure

```text
.
├── mlLearningAgents.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Use

This repository does **not** include the full Pacman framework.

To run the agent:

1. Place `mlLearningAgents.py` inside the provided Pacman coursework environment.
2. Ensure the environment contains:

   * `pacman.py`
   * `game.py`
   * `api.py`
   * `sampleAgents.py`
   * the required layout files
3. Run the Q-learning agent using:

```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```

This trains the agent for 2000 episodes and then evaluates it for 10 non-training games.

---

## Coursework Environment

The Pacman environment is based on the UC Berkeley AI Pacman framework and was adapted for the King’s College London coursework environment.

Only `mlLearningAgents.py` is included in this repository because the full Pacman framework is starter/coursework code.

---

## What I Learned

Through this coursework project, I practised:

* implementing Q-learning from scratch
* understanding exploration vs exploitation
* applying epsilon-greedy action selection
* updating Q-values using rewards and future value estimates
* training an agent through repeated game episodes
* separating training behaviour from evaluation behaviour
* integrating reinforcement learning into an existing game environment
* working within coursework constraints and an existing codebase

---

## Future Improvements

Possible improvements include:

* adding more detailed training logs
* plotting reward and win-rate over episodes
* experimenting with different `alpha`, `gamma`, and `epsilon` values
* improving state representation
* testing generalisation on more layouts
* adding comparison with SARSA
* creating a standalone simplified grid-world demo

---

## Author

**Amir Lorvand**

MSc Artificial Intelligence student at King’s College London
