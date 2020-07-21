# Reinforcement Learning Projects
This repository contains projects related to reinforcement learning.

## Easy-21 Game
Implements **monte carlo** and **sarsa-lambda** action value function update along with game simulation
environment, plotting value function and reporting mean squared error in Python.
The learnt action value function
is then used to derive the optimal playing strategy for the player against a given (but hidden from
player) dealer strategy.

The Easy-21 Game is similar to Blackjack where dealer and player start with an initial card and each
have 2 available moves at every game step : hit (get a new card, can be a +ve value or -ve value card)
or stick (yield chance to the other
player). If the sum for either player goes above 21 or below zero, they lose the game. The person with
larger sum wins the game after both stick. Detailed game description can be found at
*easy_21_game/easy-21 assignment.pdf*.

The following features are implemented:
- Game Simulation Environment.
- Dealer Strategy.
- Monte-carlo action value function update.
- Sarsa-lambda action value function update.
- epsilon-greedy exploration strategy.
- Plots action value function at end of simulation.
- Reports and plots Mean squared error between Monte-carlo and sarsa-lambda (as a function of lambda).

Sample execution cmds:

       # Run Monte-carlo action value function update
       python3 rl_easy_21.py --num-episodes=100000 --policy=monte_carlo

       # Run sarsa-lambda action value function update
       python3 rl_easy_21.py --num-episodes=100000 --policy=monte_carlo --sarsa_lambda_value=0.1

       # Run sweep of lambda with sarsa-lambda
       python3 rl_easy_21.py --num-episodes=100000 --run-sarsa-lambda-sweep=True

