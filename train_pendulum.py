# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:35:20 2023

@author: clemm
"""
from algorithms import PolicyIteration, ValueIteration, SARSA, Q_learning, TD0
import numpy as np 
import matplotlib.pyplot as plt 
import discrete_pendulum 



def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)), env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise Exception(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    print('test_x_to_s: passed')
    
def plot_traj(env, policy, name): 
    
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    rew = 0 
    while not done:
        
        a = np.argmax(policy[s,:])

        (s, r, done) = env.step(a)
        rew += r
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(rew)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    # Plot data and save to png file
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(log['t'], log['s'])
    ax[0].set_ylabel("State")
    ax[1].plot(log['t'][:-1], log['a'])
    ax[1].set_ylabel("Action")
    ax[2].plot(log['t'][:-1], log['r'])
    ax[2].set_ylabel("Cumulative Reward")
    #ax[0].legend(['s', 'a', 'r'])
    ax[3].plot(log['t'], log['theta'])
    ax[3].plot(log['t'], log['thetadot'])
    ax[3].legend(['theta', 'thetadot'])
    
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])

    fig.suptitle("Pendulum: " + str(name) + " Sample Trajectory")
    fig.savefig('./figures/pendulum/sample_traj_' + str(name) +'.png')
 
def main(): 

    env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)
    test_x_to_s(env)
    
    # the different epsilon and alpha values for SARSA and Q-Learning
    E = [0.1, 0.25, 0.5, 0.75, 0.95]
    A = [1e-3, .01, .1, .5]
    iters = np.linspace(0, 100, 100)
    
    #SARSA
    fig3, ax3 = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    for j in range(len(E)): 
        alg = SARSA(env, eps = E[j], alpha = .1, discount = .95, iters = 800)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        ax3[0].plot(iters, r, label = "$\epsilon$ = " + str(E[j]))
        ax3[1].plot(iters, epsilon)
        
    ax3[0].legend()
    ax3[1].set_xlabel("Training Episode")
    ax3[0].set_ylabel("Avg. Reward")
    ax3[1].set_ylabel(r'$\epsilon$')
    ax3[0].set_xticks([])
    ax3[0].set_ylim([-20, 60])

    fig3.suptitle( 'Pendulum: SARSA '+ r'$\alpha = 0.1$')
    fig3.savefig('./figures/pendulum/SARSA_eps_training.png')

    fig4 = plt.figure() 
    for w in range(len(A)): 
        alg = SARSA(env, eps = .95, alpha = A[w], discount = .95, iters = 800)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        plt.plot(iters, r, label = r"$\alpha$ = " + str(A[w]))
        
    plt.legend()
    plt.xlabel("Training Episode")
    plt.ylabel("Avg. Reward")
    plt.ylim([-20, 60])
    plt.title( 'Pendulum: SARSA '+ r'$\epsilon = 0.95$')
    fig3.savefig('./figures/pendulum/SARSA_alpha_training.png')


    alg = SARSA(env, eps = 0.95, alpha = .1, discount = .95, iters = 800)
    
    Q_SARSA, r, i, SARSA_eps = alg.train()
    plot_traj(env, Q_SARSA, "SARSA")
    
    TD_SARSA = TD0(env, policy = Q_SARSA, alpha = .1, discount = 0.95)
    SARSA_value = TD_SARSA.train()
    
    states = np.linspace(0, env.num_states, env.num_states)
    

    #Q-Learning 
    fig6, ax6 = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    
    for j in range(len(E)): 
        alg = Q_learning(env, eps = E[j], alpha = .1, discount = .95, iters = 800)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        ax6[0].plot(iters, r, label = "$\epsilon$ = " + str(E[j]))
        ax6[1].plot(iters, epsilon)
        
    ax6[0].legend()
    ax6[1].set_xlabel("Training Episode")
    ax6[0].set_ylabel("Return")
    ax6[1].set_ylabel(r'$\epsilon$')
    ax6[0].set_xticks([])
    ax6[0].set_ylim([-20, 60])

    fig6.suptitle( 'Pendulum: Q-Learning '+ r'$\alpha = 0.1$')
    fig3.savefig('./figures/pendulum/Q_eps_training.png')


    Q_alpha_plot = plt.figure()
    
    for w in range(len(A)): 
        alg = Q_learning(env, eps = .95, alpha = A[w], discount = .95, iters = 800)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        plt.plot(iters, r, label = r"$\alpha$ = " + str(A[w]))
        
    plt.legend()
    plt.xlabel("Training Episode")
    plt.ylabel("Return")
    plt.ylim([-20, 70])
    plt.title( 'Pendulum: Q-Learning '+ r'$\epsilon = 0.95$')
    plt.savefig('./figures/pendulum/Q_alpha_training.png')

    alg = Q_learning(env, eps = .95, alpha = .1, discount = 0.95, iters = 800)
    Q_learn, r, i, Q_eps = alg.train()
    

    plot_traj(env, Q_learn, "Q-Learning")
    
    TD_Q = TD0(env, policy = Q_learn, alpha = .1, discount = 0.95)
    Q_value = TD_Q.train()
    
    states = np.linspace(0, env.num_states, env.num_states)
    
    
    
    #Plotting the policies
    sarsa_action = []
    q_action = []
    pi_action = []
    vi_action = []
    
    for i in range(env.num_states): 
        sarsa_action.append(np.argmax(Q_SARSA[i,:]))
        q_action.append((np.argmax(Q_learn[i,:])))
       
        
    policies = plt.figure()
    plt.scatter(states, sarsa_action, color = 'green', marker = '.', label = "SARSA")
    plt.scatter(states, q_action, color = 'blue', marker = '*', label = "Q-Learning")
    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title('Pendulum: Policies')
    plt.savefig('./figures/pendulum/Policies.png')

    #Plotting value functions
    Values = plt.figure()
    plt.scatter(states, SARSA_value,  color = 'green', marker = '.', label = "SARSA")
    plt.scatter(states, Q_value,color = 'blue', marker = '*', label = "Q-Learning")
    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.title("Pendulum: Value Functions")
    plt.savefig('./figures/pendulum/ValueFunc.png')

    return 
    
    

if __name__ == '__main__':
    
    main()
    
    
