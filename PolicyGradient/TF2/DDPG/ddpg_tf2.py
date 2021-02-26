import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
            gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, 
            fc1=400, fc2=300, batch_size=64, noise=0.1):
        """
        Input parameters
        :param input_dims: input dimensions
        :param alpha: learning rate actor
        :param beta: learning rate critic
        :param env: environment
        :param gamma: discount factor for the update equation
        :param n_actions:
        :param max_size: max size for replay buffer
        :param tau: default value of the soft update (this 0.005 is taken from the paper)
        :param fc1: number of neurons in fully connected layer 1
        :param fc2: number of neurons in fully connected layer 2
        :param batch_size: batch size
        :param noise: noise parameter which reflects the std deviation of a normal distribution
        """

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions) # instantiate the ReplayBuffer
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0] #upper bound of your action space (typically +1 in continuous space)
        self.min_action = env.action_space.low[0] #lower bound on your action space (typically -1 in continuous space)

        # Instantiate the actual networks (and give them the corresponding names
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        # Compile the networks (set the optimizer and learning rate)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))

        # For the following target networks, we won't be doing gradient descent for these networks because
        # we will do the soft updates, but we still need to compile the networks (feature of tf)
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1) # hard copy of the weights to target networks (the first iteration, tau=1)

    def update_network_parameters(self, tau=None):
        # use the default parameter when tau is not specified
        if tau is None:
            tau = self.tau

        # Update the weights of the target_actor as specified below (using tau parameter)
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        # Update the weights of the target_critic
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        """Interface function which makes it clean to store the state,
        action, reward, new state, terminal state in the buffer"""
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        """
        Takes as input an observation for which we should choose an action, the evaluation parameter
        is related to whether or not your training or testing your agent. If you test the agent, you
        don't necessarily want to add noise for exploration which you want to do during the training.
        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        # Get the actions out of the network
        actions = self.actor(state)

        # if we are training, we want to get some random normal noise on our actions
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        # note that if the action output from the network is for example 1 and we add some noise to this,
        # it could be that the action value is larger than 1 while our actions should lie in the range of [-1, 1]
        # in order to fix this, we have to clip the actions by it's min and max value
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        # Return the 0th element because actions is a tensor and the value is the 0th element which is numpy array
        return actions[0]

    def learn(self):

        # What if batch size is larger than our memory (we haven't filled up our memory yet)
        # We don't really want to learn yet
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample our memory
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # Convert these numpy arrays to tensors
        states = tf.convert_to_tensor(state, dtype=tf.float32) #current states
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32) #next states when taking the action
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        # we don't need to convert the terminal flags to tensors because we will typically
        # just do some numpy array operations on it

        # We use the gradienttape and the basic idea is that we go to load up operations onto
        # our computational graph for calculation of gradients. So when we call the choose_action
        # function above, those operations aren't stored anywhere that is used for calculation of gradients
        # Only things within this context manager are used for calculation of the gradients
        # Check the formulas for the following part of the code!
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            # for the following, we have to squeeze along the first dimension because we have the batch dimension
            # and it doesn't learn if you put in the batch dimension
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            #what is the estimate of the value of the state, action pairs we encounter in the replay buffer
            critic_value = tf.squeeze(self.critic(states, actions), 1)

            # the (1-done) is 1 - true or false so when the episode is over, this is becomes 0 and you just get the rewards
            # Calculate the targets
            target = reward + self.gamma*critic_value_*(1-done)
            # Calculate the MSE loss for the critic
            critic_loss = keras.losses.MSE(target, critic_value)

        # outside the context manager, we want to calculate the gradients of the critic network.
        # We didn't define the trainable_variables variable but this is defined in Keras
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)

        # apply our gradients (update the weights of the critic network)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            # it is negative, because we are doing gradient ascent and in policy gradient methods you don't want
            # to do gradient descent because that minimizes the total score over time and we want to maximize the
            # total score over time (and gradient ascent is just the negative of the gradient descent)
            actor_loss = -self.critic(states, new_policy_actions)
            # our loss is the reduced mean of the actor loss (just computes the mean of the batch)
            actor_loss = tf.math.reduce_mean(actor_loss)

        # This is how we will the get the gradient of the critic loss with respect to the mu parameters of theta super mu.
        # This is done by taking this actor loss which is proportional to the output of the critic network and that is coupled.
        # The gradient is non-zero because it has this dependency on the output of our actor network so their dependence
        # that gives you a nonzero gradient comes from the fact that we are taking actions with respect to the actor network
        # which is calculated according to theta super mu and that gets fed forward through the critic network and that's what
        # allows you to take the gradient of the output of the critic network with respect to the variables of the actor network
        # That's how we get that coupling. If you read the paper, they apply the chain rule and you get the gradient of the critic
        # network and the gradient of the actor network but this form here is easier to implement in code.
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

        # Apply the gradients (zip up the gradient)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        # Once we have updated our main networks, we want to do the softupdate of our target networks. As it is not the first time
        # We're updating them, we just use the default value for tau
        self.update_network_parameters()
