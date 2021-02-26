import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from networks import ActorNetwork, CriticNetwork
from replaybuffer import ReplayBuffer
import numpy as np

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=1000,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions) #initialize replay buffer
        self.batch_size = batch_size
        self.learn_step_cntr = 0 #we do the delayed part of TD3, the actor network updates are delayed wrt critic updates
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        #Define all the 6 networks
        self.actor = ActorNetwork(layer1_size, layer2_size, 
                                    n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(layer1_size, layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size,name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size, 
                                    n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, name='target_critic_2')

        #Compile all the networks (setting the optimizer, initial learning rate and loss)
        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), 
                                  loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1) #Is set to 1 when we initialize the target networks during the first iteration

    def choose_action(self, observation):
        #During the warmup, just choose a random action
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0] # returns a batch size of 1, want a scalar array

        #Add some random normal noise to the chosen action
        mu_prime = mu + np.random.normal(scale=self.noise)

        #Clip the values such that the actions are within the boundaries of the min and max action
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def remember(self, state, action, reward, next_state, done):
        """Store the state, action, reward, next state in the replay buffer"""
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        #If the number of samples in the memory is smaller than batch size, do nothing
        if self.memory.mem_cntr < self.batch_size:
            return

        #Sample the replay buffer
        states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(self.batch_size)

        #Convert the numpy arrays to tensors if we want to use them in the neural networks
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        #If you use two different applied gradients for a single tape, you need to pass the persisten=True argument otherwise you don't need
        #it which means that you just have a single network that you are performing an update on
        #Read documentation https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape(persistent=True) as tape:
            #Use target actor to determine actions for those next states (s') and add some noise.
            target_actions = self.target_actor(next_states)
            target_actions = target_actions + \
                    tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action, 
                                          self.max_action)

            #Plug those actions into the target critic networks
            q1_ = self.target_critic_1(next_states, target_actions)
            q2_ = self.target_critic_2(next_states, target_actions)

            #Squeeze the output because the shape is batch size by 1 and e want to collapse this to batch size
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            #Get the min target value from the two networks
            critic_value_ = tf.math.minimum(q1_, q2_)

            #Target is the y from our paper (done flag means whether or not the episode terminated)
            target = rewards + self.gamma*critic_value_*(1-dones)

            #critic_1_loss = tf.math.reduce_mean(tf.math.square(target - q1))
            #critic_2_loss = tf.math.reduce_mean(tf.math.square(target - q2))
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        #Calculate the gradients
        critic_1_gradient = tape.gradient(critic_1_loss, 
                                          self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, 
                                          self.critic_2.trainable_variables)

        #Backpropagate the gradients to calculate the gradient of some target (often a loss) relative to some source (often the model's variables)
        self.critic_1.optimizer.apply_gradients(
                       zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
                       zip(critic_2_gradient, self.critic_2.trainable_variables))
        
        self.learn_step_cntr += 1

        #Only update the actor network evey 2 time steps (based on the update_actor_iter)
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        #Here, we're dealing with one loss, so we don't have to call the persistent=True argument
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            #It is negative because we want to maximize the return and thus this means we want to minimize the negative return
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        #This is how we get the gradient of the output of one network with respect to the parameters of another network
        # Same as applying the chain rule to the loss of the output of the critic network with respect to the parameters of the actor network
        #Calculate the gradient
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

        #Backpropagate the gradients
        self.actor.optimizer.apply_gradients(
                        zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """Update the network parameters"""

        #Is used during the first iteration such that the target networks get the same parameters of the normal networks (hard update)
        if tau is None:
            tau = self.tau

        #Update the target_actor weights
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        #Update the target_critic_1 weights
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        #Update the target_critic_2 weights
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)


