#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from scipy.signal import convolve2d
#from IPython.display import clear_outpu


# In[2]:


#tf.__version__
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)


# # Creating the game/env

# In[3]:


class connectMNK():
    def __init__(self,m,n,k):
        self.board = np.zeros((m,n), dtype = 'float32')
        self.player = np.array(1,dtype = 'float32')
        self.m = m
        self.n = n
        self.k = k
        self.history = []

    def reset(self):
        self.board = np.zeros((self.m,self.n), dtype = 'float32')
        self.player = np.array(1,dtype = 'float32')
        self.history = []
    
    def get_input_planes(self):
        '''
        r = np.array([(self.board**2+self.board)/2,
                      (self.board**2-self.board)/2,
                      np.ones((self.m,self.n)) if self.player == 1 else np.zeros((self.m,self.n)),
                      np.ones((self.m,self.n)) if self.player == -1 else np.zeros((self.m,self.n))])
        '''
        b = self.board*self.player
        r = np.array([(b**2+b)/2,
                      (b**2-b)/2])
        return np.reshape(r,(2,self.m,self.n))
    
    def valid_actions(self):
        idx = np.array(range(self.m*self.n))
        return idx[self.board.flatten()==0]
    
    def mask_actions(self):
        return (np.ones((self.m,self.n))-self.board**2).flatten()

    def do_action(self,pos):
        i = pos//self.m
        j = pos%self.n
        self.board[i][j] = self.player
        self.player = self.player * (-1)
        self.history.append(pos)
        return
        
    def undo(self):
        self.board[self.history[-1]] = 0
        self.player = self.player * (-1)
        self.history.pop()
        return

    def check_victory(self):
        horizontal_kernel = np.ones((1,self.k))
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.k, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        
        for kernel in detection_kernels:
            conv = np.array(convolve2d(self.board, kernel, mode="valid"))
            
            if (conv == self.k).any():
                return True,1
            elif (conv == -self.k).any():
                return True,-1
            
        if np.sum(self.mask_actions()) == 0:
            return True, 0
        
        return False,0
    
    def __repr__(self):
        b = self.board.astype("int")
        s=[" ","X","O"]
        r = ""
        for i in range(self.m):
            for j in range(self.n):
                r += s[b[i][j]]
                if(j<self.n-1):
                    r += "|"
            if(i<self.m-1):
                r +="\n"+"-"*(self.n*2-1)+"\n"
        return r
    
    def __str__(self):
        return self.__repr__()


# In[4]:


#For gomoku:
#M = 15
#N = 15
#K = 5

#For tic-tac-toe:
M = 3
N = 3
K = 3

game = connectMNK(M,N,K)


# In[5]:


game.reset()
moves = game.valid_actions()
game.do_action(np.random.choice(moves))
game.check_victory()


# In[6]:


done = 0
game.reset()
game.do_action(1)
game.do_action(8)
game


# # Combined Model

# In[7]:


def combined_model(blocks=1):
    input_shape = (2,M,N,)
    input1 = layers.Input(input_shape)
    
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                               use_bias=False, data_format="channels_first")(input1)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    for i in range(1,blocks):
        x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                   use_bias=False, data_format="channels_first")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    
    x = layers.Flatten()(x)
    
    probs = layers.Dense(M*N)(x)
    probs = layers.Softmax()(probs)
    
    value = layers.Dense(1,activation="tanh")(x)
    
    return tf.keras.Model(input1,[probs,value])


# In[8]:


model = combined_model(1)
model.summary()


# In[9]:


@tf.function
def model_wrap(x):
    return model(x)


# # MCTS

# In[10]:


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


# In[11]:


class Node():
    def __init__(self,prior, state, player):
        self.prior = prior
        self.state = state
        self.player = player
        
        self.value_sum = 0
        self.visit_count = 0
        self.children = {}
        
        return
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum/self.visit_count
    
    def select_action(self, temperature=1):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    
    def expand(self,action_probs):
        for i,a in enumerate(action_probs):
            if(a!=0):
                next_state = self.state.copy()
                
                j = i//game.m
                k = i%game.n
                next_state[j][k] = self.player

                self.children[i] = Node(a,next_state, self.player*(-1))
            
    def __repr__(self):
        return str(self.value())+"/"+str(self.visit_count)
    


# In[12]:


def mcts(root, simulations=5):
    for i in range(simulations):
        leaf = root
        search_path = [leaf]
        
        while leaf.children:
            action, leaf = leaf.select_child()
            search_path.append(leaf)
        
        game.board = leaf.state
        game.player = leaf.player
               
        status, value = game.check_victory()
        value = -1*abs(value)
        
        if not status:
            inputs = game.get_input_planes()
            inputs = tf.expand_dims(inputs,0)
            
            action_probs, value = model_wrap(inputs)
            
            action_probs = np.array(action_probs[0])
            
            action_probs = action_probs * game.mask_actions()
            action_probs = action_probs/np.sum(action_probs)
            
            leaf.expand(action_probs)
        
        game.board = root.state
        game.player = root.player
        
        #backpropagating values
        for node in search_path[::-1]:
            node.value_sum += float(value) if node.player == leaf.player else -float(value)
            node.visit_count += 1
  
    return root


# In[13]:


def run_episode_mcts(runs=1,simulations_per_move=5,temperature=1): 
    boards = []
    action_probs = []
    rewards_q = [] #uses the MCTS Q-value for each move
    rewards_z = [] #0 for tie, -1 for losing and +1 for winning
    q=1 #rewards_q weight
    z=0 #rewardz_z weight
    
    for i in range(runs):
        game.reset()
        
        #print('.',end="")
        root = Node(0,game.board,game.player)
        
        size = game.m*game.n
        
        for j in range(size):
            root = mcts(root, simulations_per_move)
            
            action = root.select_action(temperature)

            action_probs_t = np.zeros(size)
            for k, v in root.children.items():
                action_probs_t[k] = v.visit_count
            action_probs_t = action_probs_t / np.sum(action_probs_t)

            action_probs.append(action_probs_t)
            rewards_q.append(root.value())
            boards.append(game.get_input_planes())

            root = root.children[action]
            game.do_action(action)

            status, winner = game.check_victory()
            if status:
                break
        
        reward = [winner*(-1)**x for x in range(j+1)]
        rewards_z.extend(reward)
      
    boards = np.array(boards, dtype = "float32")
    action_probs = np.array(action_probs, dtype = "float32")
    rewards_q = np.array(rewards_q, dtype = "float32")
    rewards_z = np.array(rewards_z, dtype = "float32")
    
    rewards = q*rewards_q + z*rewards_z
    return boards, action_probs, rewards


# # Loss

# In[14]:


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
@tf.function
def compute_loss(action_probs, reward, policy, values):
    
    actor_loss = tf.reduce_mean(-(action_probs*tf.math.log(policy)))
    
    critic_loss =  tf.reduce_mean(huber_loss(reward, values))

    return actor_loss + critic_loss


# # Train Step

# In[15]:


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

@tf.function
def train_step_mcts(boards, action_probs, reward):
    with tf.GradientTape(persistent=True) as tape:
        # Calculating 
        policy, value = model(boards)

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, reward, policy, value)

    grads = tape.gradient(loss,model.trainable_variables)
    grads = tf.clip_by_global_norm(grads,5)[0]
    optimizer.apply_gradients(zip(grads,model.trainable_variables))

    return loss


# # Agents

# In[16]:


class rand():
    def __init__(self):
        return
    
    def select_action(self):
        moves = game.valid_actions()
        return np.random.choice(moves)

    
class pred():
    def __init__(self,exploit=True):
        self.exploit = exploit
        return 
    
    def select_action(self):
        inputs = game.get_input_planes()
        inputs = tf.expand_dims(inputs,0)
        
        action_probs,_ = model_wrap(inputs)

        valid_probs = action_probs * game.mask_actions()
        valid_probs = valid_probs/tf.reduce_sum(valid_probs)

        if(self.exploit):
            action = np.argmax(valid_probs)
        else:
            action = np.random.choice(list(range(len(valid_probs))),p=valid_probs)

        return action

class perf():
    ###Almost-perfect player for tic-tac-toe
    
    def __init__(self):
        return
    
    def select_action(self):
        assert game.m == 3 and game.n==3 and game.k==3
        
        b = game.board.flatten()
        player = game.player
        pos = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]

        for x in pos:
            if(np.sum(b[x])==player*2):
                for y in x:
                    if(b[y]==0):
                        return y
        for x in pos:
            if(np.sum(b[x])==player*(-2)):
                for y in x:
                    if(b[y]==0):
                        return y

        if(b[4]==0):
            return 4
    
        actions = game.valid_actions()
        corners = list(set(actions).intersection({0,2,6,8}))
        if(corners):
            return np.random.choice(corners)
        return np.random.choice(actions)
    
    
class mcts_agent():
    def __init__(self,simulations=10,temperature=0):
        self.root = Node(0,game.board,game.player)
        self.simulations = simulations
        self.temperature = temperature
        return
    
    def select_action(self):
        
        if(self.root.children):
            self.root = self.root.children[game.history[-1]]
        else:
            self.root = Node(0,game.board,game.player)
        self.root = Node(0,game.board,game.player)      
        self.root = mcts(self.root, self.simulations)
        action = self.root.select_action(self.temperature)
        self.root = self.root.children[action]
        return action
    
class human():
    def _init_(self):
        return
    
    def select_action(self):
        #clear_output()
        print("Current board")
        print(game)
        print("Available moves")
        print(game.valid_actions())
        action = input()
        return int(action)
    
def single_game(p1,p2,verbose=False):
    game.reset()
    players = [p1,p2]
    p = 0
    
    while not game.check_victory()[0]:
        if(verbose):
            print(".")
            print(game)
        move = players[p].select_action()
        game.do_action(move)

        p = 1 - p

    if(verbose):
        print(".")
        print(game)
    
    return game.check_victory()[1]


# # Train Loop

# In[17]:


max_episodes = 10
BATCH_SIZE = 128

def testing(p1=pred(),p2=rand(),runs=10):
    r1 = np.zeros(3)
    r2 = np.zeros(3)

    for k in range(runs):
        winner = int(single_game(p1, p2))
        r1[winner]+=1
        winner = int(single_game(p2, p1))
        r2[winner]+=1
    return r1/runs,r2/runs


# In[ ]:

#simulations batch_size temperature learning_rate episodes model_struct q&z 
for i in range(1,max_episodes+1):
    boards, action_probs, reward = run_episode_mcts(10,20,float("inf"))
    j = 0
    
    examples_size = len(reward)
    while j < examples_size/BATCH_SIZE:
        idx = np.random.randint(examples_size, size=BATCH_SIZE)
        train_step_mcts(boards[idx], action_probs[idx], reward[idx])
        j=j+1
        
        
    if(i%2):
        print(".",end="")

    if(i%200==0):
        print("")
        print(testing())
        

# In[ ]:


def test(players,runs):
    res = np.array([np.zeros(3),np.zeros(3)])
    
    p1 = players[0]
    p2 = players[1]
    
    print("Testing")
    for k in range(runs):
        p1.root = Node(0,game.board,game.player)
        p2.root = Node(0,game.board,game.player)
        
        winner = int(single_game(p1, p2))
        res[0][winner] += 1
        
    p1 = players[1]
    p2 = players[0]
    
    for k in range(runs):
        p1.root = Node(0,game.board,game.player)
        p2.root = Node(0,game.board,game.player)
    
        winner = int(single_game(p1, p2))
        res[1][winner] += 1
        
    return res


# In[ ]:


tf.keras.models.save_model(model,"test_run")


# In[ ]:




