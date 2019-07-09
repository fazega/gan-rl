import numpy as np
import tensorflow as tf
import random
import time
import os


class Agent():
    def __init__(self):
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []

        self.act_count = 0

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self._build_model()
        self._build_train_op()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=config)
        self.initializer = tf.global_variables_initializer()
        self.sess.run(self.initializer)

        self.saver = tf.train.Saver()
        self.save_path = "./trained_agents/a2c/"
        self.load_model()


    def load_model(self):
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.sess.run(self.initializer)
            self.saver.restore(self.sess, load_path)
        except Exception as e:
            print(e)
            print("No saved model to load, starting a new model from scratch.")
        else:
            print("Loaded model: {}".format(load_path))



    def build_generator(self):
        # Inputs
        self.input_states = tf.placeholder(dtype = tf.float32, shape = [None, 24])

        with tf.variable_scope("gen", reuse=False):
            eps = tf.random_normal(
                 shape=(tf.shape(self.input_states)[0],12),
                 mean=0., stddev=1., dtype=tf.float32)
            input_states = tf.concat([self.input_states, eps], axis=1)

            net = tf.contrib.layers.fully_connected(input_states, 200, tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            net = tf.contrib.layers.fully_connected(net, 200, tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)

            output_action = tf.contrib.layers.fully_connected(net, 4, tf.nn.tanh)
        return output_action


    def build_discriminator(self, input_action, reuse=False):
        with tf.variable_scope("disc", reuse=reuse):
            input_states = tf.concat([self.input_states, input_action], axis=1)

            net = tf.contrib.layers.fully_connected(input_states, 200, tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            net = tf.contrib.layers.fully_connected(net, 200, tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)

            output_value = tf.contrib.layers.fully_connected(net, 1, activation_fn=None)
        return output_value

    def _build_model(self):
        self.actions_ph = tf.placeholder(tf.float32, (None, 4))

        self.output_action = self.build_generator()
        self.real_value = self.build_discriminator(self.actions_ph)
        self.computed_value = self.build_discriminator(self.output_action, reuse=True)


    def _build_train_op(self):
        self.real_value_ph = tf.placeholder(tf.float32, (None,))

        self.real_value_flatten = tf.reshape(self.real_value, (-1,))
        self.disc_loss = tf.reduce_mean((self.computed_value - (-200))**2 + (self.real_value_flatten - self.real_value_ph)**2)
        self.disc_loss = tf.where(tf.is_nan(self.disc_loss),0., self.disc_loss)

        self.gen_loss = -tf.reduce_mean(self.computed_value)
        self.gen_loss = tf.where(tf.is_nan(self.gen_loss),0., self.gen_loss)

        learning_rate_gen = 0.0005
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate_gen)
        self.gradients_gen = self.optimizer_gen.compute_gradients(self.gen_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="gen"))
        self.clipped_gradients_gen = [(tf.clip_by_norm(grad, 20.0), var) if (grad is not None) else (tf.zeros_like(var),var) for grad, var in self.gradients_gen]
        self.train_gen_op = self.optimizer_gen.apply_gradients(self.clipped_gradients_gen,  self.global_step)

        learning_rate_disc = 0.0005
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate_disc)
        self.gradients_disc = self.optimizer_disc.compute_gradients(self.disc_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="disc"))
        self.clipped_gradients_disc = [(tf.clip_by_norm(grad, 20.0), var) if (grad is not None) else (tf.zeros_like(var),var) for grad, var in self.gradients_disc]
        self.train_disc_op = self.optimizer_disc.apply_gradients(self.clipped_gradients_disc,  self.global_step)

    def __call__(self, state):
        return self.sess.run(self.output_action, {self.input_states: [state]})[0]

    def addToBatch(self, state, action):
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.act_count += 1
        # self.batch_values.append(reward)

    def endGame(self, score):
        for i in range(self.act_count):
            self.batch_values.append(score)
        self.act_count = 0

    def train(self, mean_score):
        if(len(self.batch_states)-self.act_count < 500):
            return
        selected_indexes = random.choices(range(len(self.batch_states)-self.act_count),k=500)
        batch_states = np.array([self.batch_states[i] for i in selected_indexes])
        batch_actions = np.array([self.batch_actions[i] for i in selected_indexes])
        batch_values = np.array([self.batch_values[i] for i in selected_indexes])

        iter_disc = 10
        # iter_gen = 10
        # for i in range(iter_disc):
        disc_loss = self.sess.run([self.train_disc_op,self.disc_loss], {
            self.input_states: batch_states,
            self.actions_ph: batch_actions,
            self.real_value_ph: batch_values
        })

        # for i in range(iter_gen):
        gen_loss = self.sess.run([self.train_gen_op,self.gen_loss], {
            self.input_states: batch_states,
        })

        if(self.train_itr % 10 == 0):
            print("Discriminator loss: %.4f\t Generator loss: %.4f\t Mean value: %.4f"%(disc_loss[1],gen_loss[1], mean_score))
            # print(batch_actions[:2])

        if(self.train_itr % 200 == 0):
            self.save_model()

    def save_model(self):
        self.saver.save(self.sess, self.save_path, global_step=self.global_step)

    @property
    def train_itr(self):
        return self.sess.run(self.global_step)
