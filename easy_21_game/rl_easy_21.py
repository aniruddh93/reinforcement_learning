
## Implements Easy 21 (i) Game Environment and (ii) RL methods (sarsa lambda, monte-carlo and sarsa-lambda with action value
## function approximation) for determining action-value and state-value functions. These functions are then used to determine the
## optimal game playing strategy.
## Also, provides options to run sweep for "lambda" param and plot state & action value functions.
##
## Author: Aniruddh Ramrakhyani

import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import getopt
import math
from functools import partial

    

class Easy21GameEnv_t:
    """Implements the environment for Easy21 Game"""

    def __init__(self):
        """Set game start condition : Both player and dealer draw a black card"""
        self.dealer_sum = abs(self.drawCard())
        self.dealer_initial_card = self.dealer_sum
        self.player_sum = abs(self.drawCard())
        self.player_initial_card = self.player_sum
        #random.seed(a=0)

    def drawCard(self):
        """Samples and returns a card from deck at random. Card value (1-10) is uniformly 
        distributed while the color distribution probability is : red(1/3) & black(2/3)"""
        value = random.randint(1, 10)
        color = random.randint(1, 30)

        if color <= 10:
            value = -value
        return value

    def step(self, player_action):
        """Simulates a forward step in Easy21 game"""
        step_result = {
            "player_sum" : self.player_sum,
            "reward" : 0,
            "terminal" : False,
            }

        if player_action == "hit":
            self.player_sum += self.drawCard()
            step_result["player_sum"] = self.player_sum

            if (self.player_sum < 1) or (self.player_sum > 21):
                step_result["reward"] = -1
                step_result["terminal"] = True

        elif player_action == "stick":
            terminal_state_reached = False
            
            while(not terminal_state_reached):
                self.dealer_sum += self.drawCard()
                
                if self.dealer_sum >= 17:
                    step_result["terminal"] = True

                    if self.dealer_sum > self.player_sum:
                        step_result["reward"] = -1
                    elif self.dealer_sum == self.player_sum:
                        step_result["reward"] = 0
                    else:
                        step_result["reward"] = 1
                    
                    terminal_state_reached = True

        else:
            sys.exit("Illegal action: " + player_action)

        return step_result






class Policy_Interface_t:
    """Defines interface for value function update policy."""
    
    def __init__(self):
        self.action_value_fn = {}
        self.num_state_visit = {}
        self.num_state_action_visit = {}

        for dealer_initial_card in range(1, 11, 1):
            self.action_value_fn[dealer_initial_card] = {}
            self.num_state_visit[dealer_initial_card] = {}
            self.num_state_action_visit[dealer_initial_card] ={}
            
            for player_sum in range(1, 22, 1):
                self.action_value_fn[dealer_initial_card][player_sum] = {}
                self.num_state_visit[dealer_initial_card][player_sum] = 0
                self.num_state_action_visit[dealer_initial_card][player_sum] = {}

                for action in ["hit", "stick"]:
                    self.action_value_fn[dealer_initial_card][player_sum][action] = 0
                    self.num_state_action_visit[dealer_initial_card][player_sum][action] = 0




    def get_state_value_matrix(self):
        """Returns state value function (as a 2-d numpy array) using a greedy policy from action value function"""
        state_value_fn = np.zeros(shape=(10, 21))

        for dealer_initial_card in range(1, 11, 1):
            for player_sum in range(1, 22, 1):
                hit_value = self.get_action_value_fn(dealer_initial_card, player_sum, "hit")
                stick_value = self.get_action_value_fn(dealer_initial_card, player_sum, "stick")

                state_value_fn[dealer_initial_card-1][player_sum-1] = hit_value
                if stick_value > hit_value:
                    state_value_fn[dealer_initial_card-1][player_sum-1] = stick_value
        return state_value_fn



    def print_value_function_update(self):
        """Pretty prints the state value function. Prints every fifth entry in state value fn."""
        self.print_start_index = self.print_start_index % 5
        sv_fn = self.get_state_value_fn()
        print_str = ""

        for dealer_initial_card in range(0, 10, 1):
            for player_sum in range(self.print_start_index, 21, 1):
                print_str = print_str + str(round(sv_fn[dealer_initial_card][player_sum], 2)) +  "<->"
            print(print_str)
            print_str = ""
        self.print_start_index = self.print_start_index + 1



    def plot_state_value_function(self):
        """Plots the state value function."""
        # Create the input data for X & Y axis                                                                                                                 
        dealer_showing = np.arange(1, 11, 1)
        player_sum = np.arange(1, 22, 1)
        ds_2d, ps_2d = np.meshgrid(dealer_showing, player_sum)
        sv_2d = self.get_state_value_matrix()
        sv_2d = sv_2d.transpose()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surface_plot = ax.plot_surface(ds_2d, ps_2d, sv_2d, linewidth=0, antialiased=False)
        ax.set_xlabel("Dealer showing ----->", fontsize=18)
        ax.set_ylabel("Player Sum ----->", fontsize=16)
        ax.set_zlabel("State Value fn ---->", fontsize=16)

        plt.show()


    def get_num_state_visits(self, dealer_initial_card, player_sum):
        """Returns the no of visits to a game state."""
        return self.num_state_visit[dealer_initial_card][player_sum]












class Monte_Carlo_Policy_t(Policy_Interface_t):
    """Runs value function updates using monte-carlo policy."""
    
    def __init__(self):
        super().__init__()


    def update_action_value_fn(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Implements monte-carlo action value function update.
        q(s,a) = q(s,a) + alpha*(reward - q(s,a)).
        alpha = 1 / N(s,a)."""

        for i in range(0, len(action_queue)):
            cur_action = action_queue[i]
            cur_player_sum = player_sum_queue[i]

            self.num_state_visit[dealer_initial_card][cur_player_sum] += 1
            self.num_state_action_visit[dealer_initial_card][cur_player_sum][cur_action] += 1

            alpha = 1 / self.num_state_action_visit[dealer_initial_card][cur_player_sum][cur_action]
            self.action_value_fn[dealer_initial_card][cur_player_sum][cur_action] += alpha * (reward - self.action_value_fn[dealer_initial_card][cur_player_sum][cur_action])


    def get_action_value_fn(self, dealer_initial_card, player_sum, action):
        return self.action_value_fn[dealer_initial_card][player_sum][action]







class Sarsa_Lambda_Policy_t(Policy_Interface_t):
    """Runs value function update using sarsa-lambda policy."""

    def __init__(self, sarsa_lambda_param, print_sarsa_lambda_step_update=False):
        super().__init__()

        self.sarsa_lambda_param = sarsa_lambda_param
        self.eligibility_trace = {}
        self.print_sarsa_lambda_step_update = print_sarsa_lambda_step_update

        for dealer_initial_card in range(1, 11, 1):
            self.eligibility_trace[dealer_initial_card] = {}
            
            for player_sum in range(1, 22, 1):
                self.eligibility_trace[dealer_initial_card][player_sum] = {}
                
                for action in ["hit", "stick"]:
                    self.eligibility_trace[dealer_initial_card][player_sum][action] = 0





    def step_update_sarsa_lambda(self, delta, dealer_initial_card, player_sum_t, action_t, action_queue, player_sum_queue,
                                 reward_t, episode_finished):
        """Update the action value function with sarsa lambda policy"""

        # Update visit count and eligibility trace
        self.num_state_visit[dealer_initial_card][player_sum_t] += 1
        self.num_state_action_visit[dealer_initial_card][player_sum_t][action_t] += 1
        self.eligibility_trace[dealer_initial_card][player_sum_t][action_t] += 1

        # adaptive learning rate
        alpha = 1 / self.num_state_action_visit[dealer_initial_card][player_sum_t][action_t]

        # Update action value function
        for i in range(0, len(action_queue)):
            player_sum_i = player_sum_queue[i]
            action_i = action_queue[i]
            self.action_value_fn[dealer_initial_card][player_sum_i][action_i] += alpha * self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] *delta
            self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] *= self.sarsa_lambda_param

            if episode_finished:
                # Reset the eligibility trace since we have reached the terminal state of episode.                                                            
                self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] = 0

        # Print action value fn update for this step of episode. 
        if self.print_sarsa_lambda_step_update:
            self.print_sarsa_lambda_step_update(dealer_initial_card = dealer_initial_card,
                                                player_sum_queue = player_sum_queue,
                                                action_queue = action_queue,
                                                reward = reward_t)




    def run_terminal_state_sarsa_lambda_update(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue):
        """Update action value function when terminal state is reached."""
        player_sum_t = player_sum_queue[len(action_queue)-1]
        action_t = action_queue[len(action_queue)-1]
        reward = reward_queue[len(action_queue)-1]
        delta_t = reward - self.action_value_fn[dealer_initial_card][player_sum_t][action_t]
        
        self.step_update_sarsa_lambda(delta = delta_t,
                                      dealer_initial_card = dealer_initial_card,
                                      player_sum_t = player_sum_t,
                                      action_t = action_t,
                                      action_queue = action_queue,
                                      player_sum_queue = player_sum_queue,
                                      reward_t = reward,
                                      episode_finished = True)




    def update_action_value_fn(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue, episode_finished):
        """Implements SARSA-Lambda action value function update.
        e(s,a) = lambda*gamma*e(s,a) + 1(s_t = s, a_t = a).
        delta_t = reward_t + gamma*q(s_t+1, a_t+1) - q(s_t, a_t).
        q(s,a) = q(s,a) + alpha*e(s,a)*delta_t.
        Here we assume gamma=1 (undiscounted).
        alpha = 1 / N(s,a)."""

        if(len(action_queue) < 1):
            print("Nothing to update as no action taken by player !!")
            return


        if episode_finished and (len(action_queue) == 1):
            self.run_terminal_state_sarsa_lambda_update(dealer_initial_card, player_sum_queue, action_queue, reward_queue)
        else:
            if(len(action_queue) < 2):
                return
            player_sum_t = player_sum_queue[len(action_queue)-2]
            action_t = action_queue[len(action_queue)-2]
            reward = reward_queue[len(action_queue)-2]
            player_sum_t_plus_1 = player_sum_queue[len(action_queue)-1]
            action_t_plus_1 = action_queue[len(action_queue)-1]
            delta_t = reward + self.action_value_fn[dealer_initial_card][player_sum_t_plus_1][action_t_plus_1] - self.action_value_fn[dealer_initial_card][player_sum_t][action_t]


            self.step_update_sarsa_lambda(delta = delta_t,
                                          dealer_initial_card = dealer_initial_card,
                                          player_sum_t = player_sum_t,
                                          action_t = action_t,
                                          action_queue = action_queue,
                                          player_sum_queue = player_sum_queue,
                                          reward_t = reward,
                                          episode_finished = False)

            if episode_finished:
                self.run_terminal_state_sarsa_lambda_update(dealer_initial_card = dealer_initial_card,
                                                            player_sum_queue = player_sum_queue,
                                                            action_queue = action_queue,
                                                            reward_queue = reward_queue)





    def print_sarsa_lambda_step_update(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Pretty print sarsa lambda action value function update for one step."""
        print("step_reward: " + str(reward))
        for i in range(0, len(action_queue)):
            player_sum_i = player_sum_queue[i]
            action_i = action_queue[i]
            eligibility_trace_i = self.eligibility_trace[dealer_initial_card][player_sum_i][action_i]
            action_value_i = self.action_value_fn[dealer_initial_card][player_sum_i][action_i]

            print_str = "e[" + str(dealer_initial_card) + "][" + str(player_sum_i) + "][" + action_i + "]=" + str(eligibility_trace_i)
            print_str += ", q:" + str(action_value_i)
            print(print_str)
        print("-----------------")


    def get_action_value_fn(self, dealer_initial_card, player_sum, action):
        return  self.action_value_fn[dealer_initial_card][player_sum][action]









class Sarsa_Lambda_Feature_Vector_Policy_t(Policy_Interface_t):
    """Runs sarsa lambda action value function update with value function approximation using feature vector."""

    def __init__(self, sarsa_lambda_param, learning_rate):
        super.__init__()
        self.feature_vector = np.zeros(36, dtype=int)
        self.weight_vector = np.random.normal(loc=0.0, scale=1.0, size=36) # draw samples from normal distribution
        self.eligibility_trace = np.zeros(36, dtype=int)
        self.sarsa_lambda = sarsa_lambda_param
        self.alpha = learning_rate
        


    def update_feature_vector(self, dealer_initial_card, player_sum, action):
        self.feature_vector = np.zeros(36, dtype=int)
        initial_card_features = []
        player_sum_features = []
        action_feature = 0

        # dealer initial card features
        if (dealer_initial_card >= 1) and (dealer_initial_card <= 4):
            initial_card_features.append(0*12)
        if (dealer_initial_card >= 4) and (dealer_initial_card <= 7):
            initial_card_features.append(1*12)
        if (dealer_initial_card >= 7) and (dealer_initial_card <= 10):
            initial_card_features.append(2*12)

        # player sum features
        if (player_sum >= 1) and (player_sum <= 6):
            player_sum_features.append(0*2)
        if (player_sum >= 4) and (player_sum <= 9):
            player_sum_features.append(1*2)
        if (player_sum >= 7) and (player_sum <= 12):
            player_sum_features.append(2*2)
        if (player_sum >= 10) and (player_sum <= 15):
            player_sum_features.append(3*2)
        if (player_sum >= 13) and (player_sum <= 18):
            player_sum_features.append(4*2)
        if (player_sum >= 16) and (player_sum <= 21):
            player_sum_features.append(5*2)

        # action features
        if action == "hit":
            action_feature = 0
        else:
            action_feature = 1

        for ic_feat in initial_card_features:
            for ps_feat in player_sum_features:
                feature = ic_feat + ps_feat + action_feature
                self.feature_vector[feature] = 1


    def get_state_value_fn(self):
        """Returns the value of current state using the current weight vector."""
        return np.dot(np.transpose(self.feature_vector), self.weight_vector)


    def get_action_value_fn(self, dealer_initial_card, player_sum, action):
        """Returns the action value for current state and action."""
        self.update_feature_vector(dealer_initial_card, player_sum, action)
        return self.get_state_value_fn()
    

    def terminal_state_update_sarsa_lambda(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue):
        """Runs Sarsa-lambda step update for terminal state using value function approximation."""
        player_sum = player_sum_queue[len(action_queue)-1]
        action = player_sum_queue[len(action_queue)-1]
        self.update_feature_vector(dealer_initial_card, player_sum, action)
        present_state_value = self.get_state_value_fn()

        self.num_state_action_visit[dealer_initial_card][player_sum][action] += 1
        self.num_state_visit[dealer_initial_card][player_sum] += 1

        reward = reward_queue[len(action_queue)-1]
        delta = reward - present_state_value
        alpha = 1 / self.num_state_action_visit[dealer_initial_card][player_sum][action]
        self.eligibility_trace = self.sarsa_lambda * self.eligibility_trace + present_state_value
        self.weight_vector = self.weight_vector + alpha * delta * self.eligibility_trace

        for ps in player_sum_queue:
            for ac in action_queue:
                 self.num_state_action_visit[dealer_initial_card][ps][ac] = 0
                 self.num_state_visit[dealer_initial_card][ps] = 0
                 self.eligibility_trace = np.zeros(36, dtype=int)
    
    
    
    def update_action_value_fn(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue, episode_finished):
        """Update action value function (represented by function approximation) using sarsa-lambda.
           q(s, a) = features(s, a) * weight_vector
           delta_t = reward_t + gamma*q(s_t+1, a_t+1) - q(s_t, a_t)
           e(s_t, a_t) = lambda*gamma*e(s_t, a_t) + features(s_t, a_t)
           weight_vector = weight_vector + (alpha * delta_t * e(s_t, a_t))
        """
        if episode_finished and (len(action_queue) == 1):
            self.terminal_state_update_sarsa_lambda(dealer_initial_card, player_sum_queue, action_queue, reward_queue)
        else:
            if (len(action_queue) < 2):
                return
            
            # Calculate present action value 
            player_sum = player_sum_queue[len(action_queue)-2]
            action = player_sum_queue[len(action_queue)-2]
            self.update_feature_vector(dealer_initial_card, player_sum, action)
            present_state_value = self.get_state_value_fn()

            # Calculate next state action value
            next_player_sum = player_sum_queue[len(action_queue)-1]
            next_action = player_sum_queue[len(action_queue)-1]
            self.update_feature_vector(dealer_initial_card, next_player_sum, next_action)
            next_state_value = self.get_state_value_fn()

            self.num_state_action_visit[dealer_initial_card][player_sum][action] += 1
            self.num_state_visit[dealer_initial_card][player_sum] += 1
            
            reward = reward_queue[len(action_queue)-1]
            delta = reward + next_state_value - present_state_value
            alpha = 1 / self.num_state_action_visit[dealer_initial_card][player_sum][action]
            self.eligibility_trace = self.sarsa_lambda * self.eligibility_trace + present_state_value
            self.weight_vector = self.weight_vector + alpha * delta * self.eligibility_trace

            if episode_finished:
                self.terminal_state_update_sarsa_lambda(dealer_initial_card, player_sum_queue, action_queue, reward_queue)

            

    def print_sarsa_lambda_step_update(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Pretty print sarsa lambda action value function update for one step."""
        print("step_reward: " + str(reward))
        for i in range(0, len(action_queue)):
            player_sum_i = player_sum_queue[i]
            action_i = action_queue[i]
            eligibility_trace_i = self.eligibility_trace[dealer_initial_card][player_sum_i][action_i]
            self.update_feature_vector(dealer_initial_card, player_sum_i, action_i)
            action_value_i = self.get_state_value_fn()

            print_str = "e[" + str(dealer_initial_card) + "][" + str(player_sum_i) + "][" + action_i + "]=" + str(eligibility_trace_i)
            print_str += ", q:" + str(action_value_i)
            print(print_str)
        print("-----------------")



class Easy21_Control:
    """Simulates monte-carlo & sarsa-lambda based policy evaluation and control."""

    def __init__(self, print_episode_update, sarsa_lambda_param, update_policy, use_value_fn_approx = False):
        self.N_not = 100
        self.print_start_index = 1
        self.episode_id = 0
        self.print_episode_update = print_episode_update
        self.update_policy_name = update_policy
        self.policy = None
        self.use_value_fn_approx = use_value_fn_approx

        if self.update_policy_name == "sarsa_lambda":
            self.policy = Sarsa_Lambda_Policy_t(sarsa_lambda_param)
        elif (self.update_policy_name == "sarsa_lambda") and (use_value_fn_approx):
            self.policy = Sarsa_Lambda_Feature_Vector_Policy_t(sarsa_lambda_param=sarsa_lambda_param, learning_rate=0.01)
        elif self.update_policy_name == "monte_carlo":
            self.policy = Monte_Carlo_Policy_t()

        


    def simulate_episode(self):
        # Create Easy21_env. This selects initial state as part of init.
        game_env = Easy21GameEnv_t()
        terminal_state_reached = False
        episode_reward = 0
        dealer_initial_card = game_env.dealer_initial_card
        player_sum = game_env.player_initial_card
        action_queue = []
        player_sum_queue = []
        reward_queue = []
        self.episode_id += 1
        
        while(not terminal_state_reached):
            action = self.select_action(dealer_initial_card, player_sum)

            action_queue.append(action)
            player_sum_queue.append(player_sum)

            game_state = game_env.step(action)
            episode_reward = game_state["reward"]
            player_sum = game_state["player_sum"]
            terminal_state_reached = game_state["terminal"]
            reward_queue.append(episode_reward)

            if self.update_policy_name == "sarsa_lambda":
                self.policy.update_action_value_fn(dealer_initial_card = dealer_initial_card, 
                                            player_sum_queue = player_sum_queue,
                                            action_queue = action_queue,
                                            reward_queue = reward_queue,
                                            episode_finished = terminal_state_reached)

        if self.update_policy_name == "monte_carlo":        
            self.policy.update_action_value_fn(dealer_initial_card = dealer_initial_card, 
                                        player_sum_queue = player_sum_queue,
                                        action_queue = action_queue,
                                        reward = episode_reward)

        if self.print_episode_update:
            self.print_episode_summary(dealer_initial_card = dealer_initial_card,
                                       player_sum_queue = player_sum_queue,
                                       action_queue = action_queue,
                                       reward = episode_reward)




    def print_episode_summary(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Prints the summary for the episode"""
        
        # Print episode details
        print_str = "e" + str(self.episode_id) + ":"
        print_str += " episode_reward:" + str(reward)
        print(print_str)

        # Print updated action value fn
        for i in range(0, len(action_queue)):
            cur_action = action_queue[i]
            cur_player_sum = player_sum_queue[i]
            q_s_a = self.policy.get_action_value_fn(dealer_initial_card, cur_player_sum, cur_action)
            print_str = "(dealer_initial_card:" + str(dealer_initial_card) + ", cur_player_sum:" + str(cur_player_sum) 
            print_str += ", cur_action:" + cur_action + ", q:" + str(q_s_a) + ")"
            print(print_str)
        print("")

    
    def select_action(self, dealer_initial_card, player_sum):
        # With prob. (e/2 + (1-e)), select the greedy action
        # select the other action with prob. (e/2).
        # e = N_not / (N_not + N(s))

        greedy_action = "hit"
        greedy_action_value = self.policy.get_action_value_fn(dealer_initial_card, player_sum, greedy_action)
        
        non_greedy_action = "stick"
        non_greedy_action_value = self.policy.get_action_value_fn(dealer_initial_card, player_sum, non_greedy_action)
        if (non_greedy_action_value > greedy_action_value):
            greedy_action = "stick"
            non_greedy_action = "hit"

        num_state_visits = self.policy.get_num_state_visits(dealer_initial_card, player_sum)
        e = self.N_not / (self.N_not + num_state_visits)

        if (self.update_policy_name == "sarsa_lambda") and (self.use_value_fn_approx):
            e = 0.05
        
        non_greedy_action_prob = e/2
        prob = random.random()
        
        action = greedy_action
        if prob < e/2:
            action = non_greedy_action
        return action



    
    def run_control(self, num_episodes, mse_calculation_cadence=0, mse_calculation_fn=None):
        """Runs specified control for the game and returns the action-value function."""
        
        # Stores Mean Squared Error (MSE) per episode
        mse_arr = np.zeros(num_episodes, dtype=float)

        if mse_calculation_cadence > 0:
            mse_arr.resize(int(num_episodes/mse_calculation_cadence))        

        for i in range(0, num_episodes):
            self.simulate_episode()
            
            if (mse_calculation_cadence > 0) and (i % mse_calculation_cadence == 0):
                index = int(i / mse_calculation_cadence)
                mse_arr[index] = mse_calculation_fn(self.policy)
        return self.policy.get_action_value_fn, mse_arr


    def plot_state_value_function(self):
        self.policy.plot_state_value_function()







def run_policy(policy, print_episode_update, sarsa_lambda_param, num_episodes, plot_value_fn, 
               mse_calculation_cadence = 0, mse_calculation_fn = None, use_value_fn_approx = False):
    """Runs specified policy for Easy21 game and returns action value function. Plots Mean squared """

    if not (policy == "monte_carlo" or policy == "sarsa_lambda"):
        print("Invalid policy: " + policy + ". Valid policies: sarsa_lambda, monte_carlo")
        sys.exit(2)

    easy21_rl = Easy21_Control(print_episode_update=print_episode_update,
                               sarsa_lambda_param = sarsa_lambda_param,
                               update_policy=policy,
                               use_value_fn_approx = use_value_fn_approx)

    action_value_fn, mse_arr = easy21_rl.run_control(num_episodes=num_episodes,
                                                     mse_calculation_cadence = mse_calculation_cadence, 
                                                     mse_calculation_fn = mse_calculation_fn)

    if plot_value_fn:
        print("Simulation finished !! Plotting state value function!")
        easy21_rl.plot_state_value_function()

    return action_value_fn, mse_arr




def compute_mse(mc_action_value_fn, target_policy=None, target_policy_action_value_fn=None):
    """Computes mean-squared error between mc and sarsa_lambda action value functions."""
    mse = 0.0
    for dealer_initial_card in range(1, 11, 1):
        for player_sum in range(1, 22, 1):
            for action in ["hit", "stick"]:
                diff = 0.0
                if target_policy_action_value_fn is None:
                    diff = mc_action_value_fn(dealer_initial_card, player_sum, action) - target_policy.get_action_value_fn(dealer_initial_card, player_sum, action)
                elif target_policy is None:
                    diff = mc_action_value_fn(dealer_initial_card, player_sum, action) - target_policy_action_value_fn(dealer_initial_card, player_sum, action)
                else:
                    sys.exit("Either of policy or action_value_fn pointer should be provided !!!")
                    
                
                mse += diff * diff
    mse /= (11*22*2)
    mse = math.sqrt(mse)
    return mse
    




def run_sarsa_lambda_sweep_and_plot_error(print_episode_update, num_episodes, plot_episode_mse_sweep = False, use_value_fn_approx = False):
    """Runs : (i) Monte-carlo value fn update, (ii) sarsa lambda value function update with lambda varying from 0.0 to 1.0 (0.1 increment),
    (iii) Plots the mean squared error of the value function computed with monte_carlo and sarsa_lambda methods &
    (iv) Plots sweep of mean squared error (mse) with episode number."""

    # for plotting
    x = np.arange(0.0, 1.1, 0.1)
    y = []
    mse_arr_values = []
    episode_x = np.arange(0, num_episodes, 1000)

    # monte-carlo action value fn
    mc_action_value_fn, _ = run_policy(policy = "monte_carlo",
                                    print_episode_update = print_episode_update, 
                                    sarsa_lambda_param = 0.0,
                                    num_episodes = 100000,
                                       plot_value_fn = False,
                                       use_value_fn_approx = use_value_fn_approx)

    for sarsa_lambda_param in np.arange(0.0, 1.1, 0.1):
        mse_calculation_cadence = 0
        mse_calculation_fn = None
        
        if plot_episode_mse_sweep and (sarsa_lambda_param == 0.0 or sarsa_lambda_param == 1.0):
            mse_calculation_cadence = 1000
            mse_calculation_fn = partial(compute_mse, mc_action_value_fn)

        sarsa_lamda_action_value_fn, mse_arr = run_policy(policy = "sarsa_lambda",
                                                          print_episode_update = print_episode_update,
                                                          sarsa_lambda_param = sarsa_lambda_param,
                                                          num_episodes = num_episodes,
                                                          plot_value_fn = False,
                                                          mse_calculation_cadence = mse_calculation_cadence,
                                                          mse_calculation_fn = mse_calculation_fn,
                                                          use_value_fn_approx = use_value_fn_approx)

        if plot_episode_mse_sweep and (sarsa_lambda_param == 0.0 or sarsa_lambda_param == 1.0):
            mse_arr_values.append(mse_arr)

        mse = compute_mse(mc_action_value_fn, None, sarsa_lamda_action_value_fn)
        print("lambda:" + str(sarsa_lambda_param) + ", mse:" + str(mse))
        y.append(mse)

    if plot_episode_mse_sweep:
        # Plots the mse w.r.t to episode number for lambda = 0.0 and 1.0
        fig_mse_episode_sweep = plt.figure()

        ax_lambda_0 = fig_mse_episode_sweep.add_subplot(2, 1, 1)
        ax_lambda_0.plot(episode_x, mse_arr_values[0])
        ax_lambda_0.set_xlabel("episode number ------------>")
        ax_lambda_0.set_ylabel("Mean squared error -------------->")
        ax_lambda_0.set_title("lambda = 0.0")

        ax_lambda_1 = fig_mse_episode_sweep.add_subplot(2, 1, 2)
        ax_lambda_1.plot(episode_x, mse_arr_values[1])
        ax_lambda_1.set_xlabel("episode number ------------>")
        ax_lambda_1.set_ylabel("Mean squared error -------------->")
        ax_lambda_1.set_title("lambda = 1.0")


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y)
    ax.set_xlabel("sarsa_lambda param -------->")
    ax.set_ylabel("Mean Squared Error -------->")
    ax.set_title("Plot of Mean squared error (between mc and sl) with lambda sweep")
    plt.show()



def main(argv):
    """Program main"""
  
    # defaults
    policy = "monte_carlo"
    num_episodes = 200000
    print_episode_update = False
    sarsa_lambda_value = 1.0
    run_sarsa_lambda_sweep = False
    plot_episode_mse_sweep = False
    use_value_fn_approx = False

    try:
        opts, args = getopt.getopt(argv,"hp:",["policy=", "num-episodes=", "print-episode-update=", "sarsa-lambda-value=", 
                                               "run-sarsa-lambda-sweep=", "plot-episode-mse-sweep=", "use-value-fn-approx="])

    except getopt.GetoptError:
        print("abc.py -p <rl update policy: sarsa_lambda, monte_carlo> --num-episodes=<no. of episodes to simulate> --print-episode-update=<True or False>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("abc.py -p <rl update policy: sarsa_lambda, monte_carlo> --num-episodes=<no. of episodes to simulate> --print-episode-update=<True or False>")
            sys.exit()
        elif opt in ("-p", "--policy"):
            policy = arg
        elif opt in ("--num-episodes"):
            num_episodes = int(arg)
        elif (opt in ("--print-episode-update")) and (arg == "True"):
            print_episode_update = True
        elif (opt in ("--sarsa_lambda_value")):
            sarsa_lambda_value = float(arg)
        elif (opt in ("--run-sarsa-lambda-sweep")) and (arg == "True"):
            run_sarsa_lambda_sweep = True
        elif (opt in ("--plot-episode-mse-sweep")) and (arg == "True"):
            plot_episode_mse_sweep = True
        elif (opt in ("--use-value-fn-approx")) and (arg == "True"):
            use_value_fn_approx = True
            


    if run_sarsa_lambda_sweep:
        run_sarsa_lambda_sweep_and_plot_error(print_episode_update, num_episodes, plot_episode_mse_sweep)
    else:
        run_policy(policy=policy,
                   print_episode_update=print_episode_update,
                   sarsa_lambda_param = sarsa_lambda_value,
                   num_episodes = num_episodes,
                   plot_value_fn = True)




if __name__ == "__main__":
    main(sys.argv[1:])
