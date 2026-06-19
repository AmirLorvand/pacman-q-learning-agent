# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        # pacman position
        self.pacman_pos = state.getPacmanPosition()

        # position of all ghosts
        self.ghost_positions = tuple(state.getGhostPositions())

        # number of remaining food pellets
        self.food_count = state.getNumFood()

        # legal actions in this state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        self.legal_actions = tuple(sorted(legal))

        # terminal flags
        self.is_win = state.isWin()
        self.is_lose = state.isLose()

    def __eq__(self, other):
        # compare two states
        return isinstance(other, GameStateFeatures) and (
            self.pacman_pos == other.pacman_pos and
            self.ghost_positions == other.ghost_positions and 
            self.food_count == other.food_count and
            self.legal_actions == other.legal_actions and
            self.is_win == other.is_win and 
            self.is_lose == other.is_lose
        )
    
    def __hash__(self):
        # save state as dictionary key
        return hash((
            self.pacman_pos,
            self.ghost_positions,
            self.food_count,
            self.legal_actions,
            self.is_win,
            self.is_lose
        ))
    
    def __repr__(self):
        return f"State({self.pacman_pos})"


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 10,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Q table: maps (state, action) to Q value
        self.qValues = {}

        # count table: maps (state, action) to visit count
        self.counts = {}

        # store previous state and action
        self.lastState = None
        self.lastAction = None

        # previous raw game state for reward computation
        self.prevGameState = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # score change after one move (reward)
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # return the Q value of (state, action)
        return self.qValues.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # use only legal actions
        legal = list(state.legal_actions)

        # return 0 if there is no available action
        if not legal:
            return 0.0
        
        # return the best Q value from this state
        return max(self.getQValue(state, action) for action in legal)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # current Q value
        oldQ = self.getQValue(state, action)

        # best future Q value
        nextMaxQ = self.maxQValue(nextState)

        # Q learning target
        target = reward + self.gamma * nextMaxQ

        # update Q value
        self.qValues[(state, action)] = oldQ + self.alpha * (target - oldQ)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # increase visit count by 1
        key = (state, action)
        self.counts[key] = self.counts.get(key, 0) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # prefer actions tried only a few times
        if counts < self.maxAttempts:
            return 1.0
        
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # legal move for the current state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # if no legal move available
        if not legal:
            return Directions.STOP
        
        # current state
        currentState = GameStateFeatures(state)

        # learn from previous transition
        if self.lastState is not None and self.lastAction is not None:
            reward = self.computeReward(self.prevGameState, state)
            self.learn(self.lastState, self.lastAction, reward, currentState)

        # exploration and exploitation
        if util.flipCoin(self.epsilon):
            #random exploration
            action = random.choice(legal)
        else:
            # pick action with higest exploration value
            bestValue = float('-inf')
            bestActions = []

            for a in legal:
                q = self.getQValue(currentState, a)
                n = self.getCount(currentState, a)
                
                # use exploration only during training
                if self.episodesSoFar < self.numTraining:
                    value = self.explorationFn(q, n)
                else:
                    value = q

                if value > bestValue:
                    bestValue = value
                    bestActions = [a]
                elif value == bestValue:
                    bestActions.append(a)
            
            action = random.choice(bestActions)
        
        # update visit count
        self.updateCount(currentState, action)

        # store transition data for next step
        self.lastState = currentState
        self.lastAction = action
        self.prevGameState = state

        return action

    def final(self, state: GameState):
        """
        Called by Pacman at the end of each game
        """
        # learn from the final transition
        if self.lastState is not None and self.lastAction is not None and self.prevGameState is not None:
            finalState = GameStateFeatures(state)
            reward = self.computeReward(self.prevGameState, state)
            self.learn(self.lastState, self.lastAction, reward, finalState)

        # reset episode memory
        self.lastState = None
        self.lastAction = None
        self.prevGameState = None

        # count completed episodes
        self.episodesSoFar += 1

        # stop learning after training
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0
            self.alpha = 0.0
