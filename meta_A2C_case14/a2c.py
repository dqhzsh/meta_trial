import os
import tensorflow as tf
from tqdm import tqdm
import warnings
import numpy as np
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
import tensorflow.keras.backend as K
import random
from grid2op.MakeEnv import make
from grid2op.Action import *
from lightsim2grid.lightSimBackend import LightSimBackend

from grid2op.Reward import *

from reward import NormalizedL2RPNReward


from trainingParam import TrainingParam
from a2c_NN import A2C


class Agent(AgentWithConverter):

    def __init__(self,
                 env,
                 observation_space,
                 action_space,
                 lr_actor=1e-5,
                 lr_critic=1e-4,
                 training_param=TrainingParam()):

        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        print("转换后动作空间大小为{}".format(self.action_space.size()))

        self.env = env
        self._training_param = training_param
        #self.observation_space = observation_space
        self.observation_size = self._training_param.get_obs_size(env, self._training_param.list_attr_obs)
        print("挑选后状态空间大小为{}".format(self.observation_size))
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        #self.A2C = A2C(self.action_space.size(), self.observation_space.size(), lr=self.lr, training_param=self._training_param)
        self.A2C = A2C(self.action_space.size(), self.observation_size, lr_actor=self.lr_actor,
                       lr_critic=self.lr_critic, training_param=self._training_param)
        self.A2C_worker = A2C(self.action_space.size(), self.observation_size,  lr_actor=self.lr_actor,
                       lr_critic=self.lr_critic, training_param=self._training_param)

    def init_training(self):
        self.epoch_rewards = []  # 存储平均每条轨迹reward
        self.epoch_alive = []  # 存储平均每条存活步数
        self.loss_actor_list = []  # 存储每轮loss
        self.loss_critic_list = []
        self.A2C = A2C(self.action_space.size(), self.observation_size, lr_actor=self.lr_actor,
                       lr_critic=self.lr_critic, training_param=self._training_param)

    def convert_obs(self, observation):
        # Made a custom version to normalize per attribute
        # return observation.to_vect()
        li_vect = []
        # for el in observation.attr_list_vect:
        for el in self._training_param.list_attr_obs:
            v = observation._get_array_from_attr_name(el).astype(float)
            # if el == 'prod_p' or el == 'load_p' or el == 'load_q':
            #     v = v/100
            # if el == 'prod_v':
            #     v = v/self.env.backend.prod_pu_to_kv
            # if el == 'hour_of_day':
            #     v = v/24
            # if el == 'minute_of_hour':
            #     v = v/60

            # v_fix = np.nan_to_num(v)
            # v_norm = np.linalg.norm(v_fix)
            # if v_norm > 1e6:
            #     v_res = (v_fix / v_norm) * 10.0
            # else:
            #     v_res = v_fix
            # li_vect.append(v_res)
            li_vect.append(v)
        return np.concatenate(li_vect)

    def convert_act(self, action):
        return super().convert_act(action)

    def my_act(self, observation, reward, done=False):
        predict_movement_int, *_ = self.A2C.predict_movement(observation.reshape(1, -1))
        return int(predict_movement_int)

    def load(self, path):
        # not modified compare to original implementation
        self.A2C.load_network(path)

    def save(self, path):
        if path is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            nm_conv = "action_space.npy"
            conv_path = os.path.join(path, nm_conv)
            if not os.path.exists(conv_path):
                self.action_space.save(path=path, name=nm_conv)

            self._training_param.save_as_json(path, name="training_params.json")
            self.A2C.save_network(path, name="A2C")

    def meta_lr(self, env, id, N_TASK=1):
        # print(envs)
        # actor_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
        # critic_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        meta_optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.MeanSquaredError()
        env_id = env
        env_id.set_id(id)
        env_id.reset()
        print(env_id.chronics_handler.real_data.get_id())
        new_obs = env_id.reset()
        state = self.convert_obs(new_obs)

        alive_steps = 0
        step_inner = 0
        episode_num = 0
        states = []  # buffer
        actions = []
        new_states = []
        dones = []
        rewards = []
        actions_probs = []
        done = False
        # 初始化零梯度列表
        meta_grads_a = [tf.zeros_like(var) for var in self.A2C.model_policy_head.trainable_variables]
        meta_grads_c = [tf.zeros_like(var) for var in self.A2C.model_critic_head.trainable_variables]

        while (step_inner <= self._training_param.max_step):
            if done:
                new_obs = env_id.reset()
                state = self.convert_obs(new_obs)
            a, a_prob = self.A2C_worker.predict_movement(state)
            act = self.convert_act(a)
            new_obs, reward, done, info = env_id.step(act)
            new_state = self.convert_obs(new_obs)

            # 记录一条轨迹
            states.append(state)
            actions.append(a)
            rewards.append(reward)
            dones.append(done)
            new_states.append(new_state)
            step_inner += 1

            state = new_state

            if done or alive_steps == self._training_param.max_step or step_inner == self._training_param.max_step:
                episode_num += 1
                states, actions, rewards, new_states, dones = np.array(states), np.array(actions), np.array(
                    rewards), np.array(new_states), np.array(dones)
                # 求GAE
                dones = tf.Variable(dones, dtype=tf.float32)
                dones = tf.expand_dims((1.0 - dones), axis=1)  # done为0，非done为1，用于筛选非done
                last_state = np.expand_dims(new_states[-1], axis=0)
                adv_batch = self.A2C_worker.get_advantages(states, dones, rewards, last_state)

                # 求V_target
                V_last = self.A2C_worker.model_critic_head.predict(last_state)
                rewards[-1] += self.A2C_worker.training_param.gama * dones[-1] * V_last
                V_target = self.A2C_worker.calculate_returns(rewards, dones)
                act_batch = tf.one_hot(actions, self.A2C_worker.action_size)
                with tf.GradientTape() as tape:
                    # actor loss
                    act_pred = self.A2C_worker.model_policy_head(states)
                    newpolicy_probs = K.sum(act_pred * act_batch, axis=1)
                    loss_actor = -(adv_batch * K.log(newpolicy_probs + 1e-10))
                    loss_actor = tf.reduce_mean(loss_actor)
                    loss_entropy = -K.sum(act_pred * K.log(act_pred + 1e-10))
                    loss_entropy = tf.reduce_mean(loss_entropy)
                    loss_actor += loss_entropy

                # 使用 tf.GradientTape 计算 Actor 网络损失 loss_actor 相对于 Actor 网络可训练变量的梯度 grads_a
                grads_a = tape.gradient(loss_actor, self.A2C_worker.model_policy_head.trainable_variables)
                #actor_optimizer.apply_gradients(zip(grads_a, self.A2C_worker.model_policy_head.trainable_variables))

                # for g, w in zip(grads_a, meta_grads_a):
                #     w += g
                for i, g in enumerate(grads_a):
                    meta_grads_a[i] += g

                with tf.GradientTape() as tape:
                    # critic loss
                    value_pred = self.A2C_worker.model_critic_head(states)
                    loss_critic = loss_func(value_pred, V_target)

                grads_c = tape.gradient(loss_critic, self.A2C_worker.model_critic_head.trainable_variables)
                #critic_optimizer.apply_gradients(zip(grads_c, self.A2C_worker.model_critic_head.trainable_variables))

                # for g, w in zip(grads_c, meta_grads_c):
                #     w += g
                for i, g in enumerate(grads_c):
                    meta_grads_c[i] += g


                # 清空缓冲区并重新初始化
                states = []
                actions = []
                new_states = []
                dones = []
                rewards = []

                alive_steps = 0

            else:
                alive_steps += 1

        meta_grads_a = [g / episode_num /N_TASK for g in meta_grads_a]
        meta_grads_c = [g / episode_num /N_TASK for g in meta_grads_c]
        meta_optimizer.apply_gradients(zip(meta_grads_a, self.A2C.model_policy_head.trainable_variables))
        meta_optimizer.apply_gradients(zip(meta_grads_c, self.A2C.model_critic_head.trainable_variables))



    def normalize_rewards(self, rewards):
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-3  # Prevent division by zero
        normalized_rewards = (rewards - mean_reward) / std_reward
        normalized_rewards = np.clip(normalized_rewards,-3,3)
        return normalized_rewards

    def random_task(self, N_task):
        tasks = []
        # Loop for N_task times
        for _ in range(N_task):
            random_env = self.env
            random_env.chronics_handler.tell_id(random.randint(0, 999))
            print("ID of chronic current folder:", random_env.chronics_handler.real_data.get_id())
            # Reset the environment
            #env.reset()
            random_env.set_id(random_env.chronics_handler.real_data.get_id())
            random_env.reset()
            tasks.append(random_env.chronics_handler.real_data.get_id())
        return tasks

    def train(self,
            env,
            train_step,
            save_path,
            logdir=None,
            training_param=None,
            verbose=True
            ):

        if training_param is None:
            training_param = TrainingParam()

        if self._training_param is None:
            self._training_param = training_param
        else:
            self.training_param = self._training_param

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        env.set_chunk_size(int(max(100, nb_ts_one_day)))#优化数据读取过程，等于下面一行的函数，读一天大小的数据
        # self._set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        states = []  # buffer
        actions = []
        new_states = []
        dones = []
        rewards = []
        actions_probs = []
        alive_steps = 0
        total_reward = 0
        step = 0 #外循环的步数
        N_task = 5
        self.init_training()

        with tqdm(total=train_step, disable=False, miniters=1, mininterval=3) as pbar:
            train_summary_writer = tf.summary.create_file_writer(logdir)
            # meta_a_w = self.A2C.model_policy_head.get_weights()
            # meta_c_w = self.A2C.model_critic_head.get_weights()

            while(step < train_step): #外循环
                meta_a_w = self.A2C.model_policy_head.get_weights()
                meta_c_w = self.A2C.model_critic_head.get_weights()

                env_list = self.random_task(N_task)
                weights_a = np.copy(meta_a_w)
                weights_c = np.copy(meta_c_w)

                for _ in env_list:   #内循环
                    step_inner = 0  # 内循环的步数
                    done = False
                    env_id = env
                    env_id.set_id(_)
                    env_id.reset()
                    print(env_id.chronics_handler.real_data.get_id())
                    new_obs = env_id.reset()
                    state = self.convert_obs(new_obs)
                    self.A2C_worker.model_policy_head.set_weights(weights_a)
                    self.A2C_worker.model_critic_head.set_weights(weights_c)
                    while(step_inner <= self._training_param.max_step):  # 设置每个环境跑max_step步数
                        if done:
                            new_obs = env_id.reset()
                            state = self.convert_obs(new_obs)
                        a, a_prob = self.A2C_worker.predict_movement(state)
                        act = self.convert_act(a)
                        new_obs, reward, done, info = env_id.step(act)
                        new_state = self.convert_obs(new_obs)

                        # 记录一条轨迹
                        total_reward += reward

                        states.append(state)
                        actions.append(a)
                        rewards.append(reward)
                        dones.append(done)
                        new_states.append(new_state)
                        actions_probs.append(a_prob)

                        step += 1
                        step_inner += 1

                        state = new_state

                        if done or alive_steps == self._training_param.max_step or step_inner == self._training_param.max_step:
                            # Normalize rewards before training
                            # normalized_rewards = self.normalize_rewards(rewards)
                            # 回合更新
                            loss_actor, loss_critic = self.A2C_worker.train(np.array(states), np.array(actions), np.array(rewards),
                                                                        np.array(new_states), np.array(dones),
                                                                        np.array(actions_probs))
                            if done:
                                self.epoch_rewards.append(total_reward)
                                self.epoch_alive.append(alive_steps)
                                self.loss_actor_list.append(loss_actor)
                                self.loss_critic_list.append(loss_critic)

                            states.clear()  # buffer
                            actions.clear()
                            new_states.clear()
                            dones.clear()
                            rewards.clear()
                            actions_probs.clear()

                            if done:
                                print("Survived [{}] steps".format(alive_steps))
                                print("Total reward [{}]".format(total_reward))

                            alive_steps = 0
                            total_reward = 0
                            if step_inner == self._training_param.max_step: #内层跑满max_step步进行元更新
                                # 进行元学习
                                self.meta_lr(env_id, _, len(env_list))
                            #break

                        else:
                            alive_steps += 1

                        if step % 100 == 0 and len(self.epoch_rewards) >= 1:
                            with train_summary_writer.as_default():
                                mean_reward = np.mean(self.epoch_rewards)
                                mean_alive = np.mean(self.epoch_alive)
                                mean_loss_actor = np.mean(self.loss_actor_list)
                                mean_loss_critic = np.mean(self.loss_critic_list)

                                if len(self.epoch_rewards) >= 20:
                                    mean_reward_20 = np.mean(self.epoch_rewards[-20:])
                                    mean_alive_20 = np.mean(self.epoch_alive[-20:])
                                    mean_loss_actor_20 = np.mean(self.loss_actor_list[-20:])
                                    mean_loss_critic_20 = np.mean(self.loss_critic_list[-20:])

                                else:
                                    mean_reward_20 = mean_reward
                                    mean_alive_20 = mean_alive
                                    mean_loss_actor_20 = mean_loss_actor
                                    mean_loss_critic_20 = mean_loss_critic

                                if len(self.epoch_rewards) >= 1:
                                    tf.summary.scalar("mean_reward_20", mean_reward_20, step)
                                    tf.summary.scalar("mean_alive_20", mean_alive_20, step)
                                    tf.summary.scalar("mean_loss_actor_20", mean_loss_actor_20, step)
                                    tf.summary.scalar("mean_loss_critic_20", mean_loss_critic_20, step)

                        pbar.update(1)

                self.save(save_path)

        # 迭代结束
        self.save(save_path)