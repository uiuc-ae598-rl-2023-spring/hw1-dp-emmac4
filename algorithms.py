# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:40:18 2023

@author: clemm
"""

import numpy as np 
import random

class PolicyIteration(): 
    def __init__(self, env, discount, theta): 
        self.discount = discount
        self.env = env 
        
        self.action_space = env.num_actions 
        self.obs_space = env.num_states
        
        #Estimation accuracy 
        self.theta = theta
        
        #Initialize value
        self.V = np.zeros(self.obs_space)
        
        return 
    
    def value_update(self, s, pi): 
        x = []
        for i in range(self.obs_space): 
            
            p = self.env.p(i, s, pi[s])
            r = self.env.r(s, pi[s])
            x.append(p*(r+self.discount*self.V[i]))
             
        new_value = np.sum(x)
        return new_value
    
    def policy_update(self, s):
        a = [] 
        
        for i in range(self.action_space): 
            x = []
            for j in range(self.obs_space): 
                p = self.env.p(j,s, i)
                r = self.env.r(s, i)
                t = p*(r + self.discount*self.V[j])
                x.append(t) 
            x_sum = np.sum(x)
            a.append(x_sum)
        
        new_policy = np.argmax(a)
        return new_policy
    
    def evaluation(self, pi): 
        delta = 0  
        for s in range(self.obs_space): 
            old_v = self.V[s]
            new_v = self.value_update(s, pi)
            delta = np.max([delta, np.abs(old_v - new_v)])
            self.V[s] = new_v
        return delta 
    
    def improvement(self, pi): 
        stable = False 
        old_action = pi
        new_action = []
        
        for s in range(self.obs_space): 
            new_action.append(self.policy_update(s))
        new_action = np.asarray(new_action)
        
        if (old_action == new_action).all() == True: 
            stable = True
        
        return stable, new_action
    
    def train(self): 
        pi = np.zeros(self.obs_space)
        iters = 0
        mean_value = []
        policy_stable = False 
        
        deltas = []
        
        while policy_stable != True: 
            d = 0.0
            #Initial value evaluation to get real delta for loop 
            #print("evalutaion iteration: " + str(iters))
            d = self.evaluation(pi)
            #print("delta = " + str(d))
            mean_value.append(np.mean(self.V))
            iters += 1
            deltas.append(d)
            while d > self.theta: 
                '''
                if iters % 10 == 0: 
                    print("evaluation iteration: " + str(iters))
                    print("delta = " + str(d))
                '''
                d  = self.evaluation(pi)
                mean_value.append(np.mean(self.V))
                iters += 1
                deltas.append(d)
            #print("policy improvement")
            policy_stable, pi = self.improvement(pi)
        
        return iters, self.V, mean_value, pi, deltas
    
class ValueIteration(): 
    def __init__(self, env, theta, discount): 
        self.discount = discount
        self.env = env 
        
        self.action_space = env.num_actions 
        self.obs_space = env.num_states
        
        #estimation accuracy
        self.theta = theta
        
        #initialize V 
        self.V = np.zeros(self.obs_space)
        return 
    
    def value_update(self,s): 
        v = []
        
        for a in range(self.action_space):
            x = self.env.r(s, a)
            for i in range(self.obs_space):
                p = self.env.p(i, s, a)
                t = p*(self.discount*self.V[s])
                x += t
                
            v.append(x)
        new_v = np.max(v)
        return new_v
            
    def evaluation(self): 
        delta = 0  
        for s in range(self.obs_space): 
            old_v = self.V[s]
            new_v = self.value_update(s)
            delta = np.max([delta, np.abs(old_v - new_v)])
            self.V[s] = new_v
        return delta 
    
    def policy(self, s, V): 
        a = [] 
        
        for i in range(self.action_space): 
            x = []
            for j in range(self.obs_space): 
                p = self.env.p(j,s, i)
                r = self.env.r(s, i)
                t = p*(r + self.discount*V[j])
                x.append(t) 
            x_sum = np.sum(x)
            
            a.append(x_sum)
        
        new_policy = np.argmax(a)
        return new_policy
          
    
    def train(self): 
        delta = 0.0
        deltas = []
        iters = 0
        mean_value = []
        
        #Initial value evaluation to get real delta for loop
        s = 0 
        delta = self.evaluation()
        iters += 1 
        s += 1
        mean_value.append(np.mean(self.V))
        deltas.append(delta)
        while delta > self.theta: 
            '''
            if iters % 10 == 0: 
                print("iteration: " + str(iters))
                print("evalutiation delta = " + str(delta))
            '''
            for s in range(self.obs_space):
                delta = self.evaluation()
                
            deltas.append(delta)   
            mean_value.append(np.mean(self.V))
            iters += 1
        
        pi = []
        for i in range(self.obs_space):   
            pi.append(self.policy(i,self.V))
        
        return iters,self.V,  mean_value, pi, deltas

class SARSA(): 
    def __init__(self, env, eps, alpha, discount, iters): 
        self.eps = eps 
        self.alpha = alpha
        self.discount = discount
        
        #number of episodes to train for
        self.iters = iters
        
        self.env = env
        self.obs_space = env.num_states
        self.action_space = env.num_actions
        
        return 
    
    def greedy_action(self, s, Q): 
        p = np.random.random()
        if p < self.eps: 
            a = random.randrange(self.action_space)
        else: 
            a = np.argmax(Q[s,:])
             
        return a 
    
    def update(self, s, a, s_new, a_new, r, Q): 
        new_Q = Q[s,a] + self.alpha*(r + self.discount*Q[s_new, a_new]-Q[s,a])
        return new_Q
    
    
    def train(self):
        #initialize Q
        Q = np.zeros([self.obs_space, self.action_space])
        
         
        
        #set epsilon decay parameters
        decay = .99 
        min_eps = 0.01
        epsilon = []
        
        rew = []
        for i in range(self.iters): 
            #initialize state
            s = self.env.reset()
            
            #get greedy action
            a = self.greedy_action(s, Q)
            
            done = False
            r_ep = []
            epsilon.append(self.eps)
            
            while not done: 
                #step through environment 
                (s_new, r, done) = self.env.step(a)
                a_new = self.greedy_action(s_new, Q)
                
                #update Q with new state and action
                Q[s, a] = self.update(s, a, s_new, a_new, r, Q)
                s = s_new 
                a = a_new
                r_ep.append(r)
             
             
            rew.append(np.sum(r_ep))
            i += 1
            if i % 10 == 0: 
                #decay epsilon every 10 iterations
                self.eps = np.max([min_eps, self.eps*decay])
            
        return Q, rew, i, epsilon
    
class Q_learning(): 
    def __init__(self, env, eps, alpha, discount, iters): 
        self.eps = eps 
        self.alpha = alpha
        self.discount = discount
        
        #training iterations
        self.iters =iters
        
        self.env = env
        self.obs_space = env.num_states
        self.action_space = env.num_actions
        return 
    
    def greedy_action(self, s, Q): 
        p = np.random.random()
        if p < self.eps: 
            a = random.randrange(self.action_space)
        else: 
            a = np.argmax(Q[s,:])
             
        return a 
    
    def update(self, s, a, s_new, r, Q): 
        new_Q = Q[s,a] + self.alpha*(r + self.discount*np.max(Q[s_new, :])-Q[s,a])
        return new_Q
    
    def train(self):
        
        
        #epsilone decay parameters
        decay = .99
        min_eps = 0.01
        epsilon = []
        
        #Initialize Q
        Q = np.zeros([self.obs_space, self.action_space])
        
        rew = []
        for i in range(self.iters): 
            #Initialize state
            s = self.env.reset()
            
            done = False 
            r_ep = []
            epsilon.append(self.eps)
            
            while not done: 
                #get greedy action
                a = self.greedy_action(s, Q)
                
                #step through environment 
                (s_new, r, done) = self.env.step(a)
                r_ep.append(r)
                
                #Update Q with new state
                Q[s,a] = self.update(s, a, s_new, r, Q)
                s = s_new 
                
            rew.append(np.sum(r_ep))
            i += 1 
            if i % 10 == 0: 
                self.eps = np.max([min_eps, self.eps*decay])
                
        return Q, rew, i, epsilon
    
class TD0(): 
    def __init__(self, env, policy, alpha, discount):
        self.pi = policy 
        
        self.alpha = alpha 
        self.discount = discount
        
        self.env = env
        self.obs_space = env.num_states
        self.action_space = env.num_actions
        return 
    
    def update(self,V, s, r, s_new): 
        new_V = V[s] + self.alpha*(r +self.discount*V[s_new] - V[s])
        return new_V
    
    def train(self): 
        #Initialize V
        V = np.zeros(self.obs_space)
        
        for i in range(300): 
            #initialize state
            s = self.env.reset()
            done = False 
            
            while not done: 
                #get action from policy
                a = np.argmax(self.pi[s, :])
                
                #step through environment
                (s_new, r, done) = self.env.step(a)
                
                #update V with policy action and new state
                new_V = self.update(V, s, r, s_new)
                
                V[s] = new_V
                s = s_new
            
        return V
        
        
        
        
        
        
        
        
        
        
