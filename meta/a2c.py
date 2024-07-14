import os
import tensorflow as tf
from tqdm import tqdm
import warnings
import numpy as np
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
import tensorflow.keras.backend as K
import random
import gc
import uuid

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

        # meta_1=self.create_meta_env()
        # meta_2=self.create_meta_env()
        # print(meta_1==meta_2)
        self.env = env
        #self.random_task(5)


        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)
        self.action_space.filter_action(self._filter_action)
        print("转换后动作空间大小为{}".format(self.action_space.size()))

        
        self._training_param = training_param
        self.observation_space = observation_space
        self.observation_size = self._training_param.get_obs_size(env, self._training_param.list_attr_obs) + env.n_line
        print("挑选后状态空间大小为{}".format(self.observation_size))
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        #self.A2C = A2C(self.action_space.size(), self.observation_space.size(), lr=self.lr, training_param=self._training_param)
        self.A2C = A2C(self.action_space.size(), self.observation_size, lr_actor=self.lr_actor,
                       lr_critic=self.lr_critic, training_param=self._training_param)
        self.A2C_worker = A2C(self.action_space.size(), self.observation_size,  lr_actor=self.lr_actor,
                       lr_critic=self.lr_critic, training_param=self._training_param)

        #self.create_meta_env()
        

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem == MAX_ELEM:
            return True
        return False

    def init_training(self):
        self.epoch_rewards = []  # 存储平均每条轨迹reward
        self.epoch_alive = []  # 存储平均每条存活步数
        self.loss_actor_list = []  # 存储每轮loss
        self.loss_critic_list = []
        self.env_list = []
        self.id = 0
        self.num = 0
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
        # return np.concatenate(li_vect)

        # evaluate the danger degree of line, and append it to line state vector as the 6th/last feature
        danger = 0.9
        self.thermal_limit_under400 = tf.Variable(self.env._thermal_limit_a < 400)
        obsrho = getattr(observation, "rho")
        danger_ = ((obsrho >= (danger - 0.05)) & self.thermal_limit_under400) | (obsrho >= danger)
        d_vect = tf.cast(danger_, dtype=float)
        li_vect.append(d_vect)

        return np.concatenate((li_vect), axis=0)

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

    def create_meta_env(self):
        meta_env = make(
            dataset = r"/nfs01/projects/50501254/s120222227104/data_grid2op/l2rpn_neurips_2020_track2_small",
            reward_class=NormalizedL2RPNReward,
            action_class=TopologyChangeAndDispatchAction,
            backend=LightSimBackend()
        )
        meta_env.action_space = self.action_space
        # print(meta_env.action_space.size())
        # print(meta_env.action_space == self.action_space)
        meta_env.observation_size = self.observation_size
        # print(meta_env == self.env)

        return meta_env

    def generate_dynamic_name(self):
        return "env_meta_" + str(uuid.uuid4()).replace('-', '')


    def meta_lr(self,env, N_TASK=1):
        print("meta start")
        # print(envs)
        meta_optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.MeanSquaredError()

        if(self.num==5):
            self.num = 0
        # env_id = self.env[env_name]
        # env_id.set_id(id)
        # env_id.reset()
        # print(env_id.chronics_handler.real_data.get_id())

        # self.env[env_name].set_id(id)
        # self.env[env_name].reset()
        # print(self.env[env_name].chronics_handler.real_data.get_id())
        # print(self.id)
        # print(self.env_list)
        # print(self.env_list[self.id])
        # print(env.chronics_handler.real_data.get_id())


        # env.reset()
        # pre_id = self.env_list[self.id]
        # dynamic_name = self.generate_dynamic_name()
        # globals()[dynamic_name] = self.create_meta_env()
        
        #env_meta = self.create_meta_env()

        # env_meta = make(#env_name,
        #        #reward_class=L2RPNReward,
        #        dataset = r"/nfs01/projects/50501254/s120222227104/data_grid2op/l2rpn_neurips_2020_track2_small",
        #        reward_class=NormalizedL2RPNReward,
        #        action_class=TopologyChangeAndDispatchAction,
        #        backend=LightSimBackend()
        #     )


        #current_id = self.env_list[self.id+1]
        #print(current_id==env.chronics_handler.real_data.get_id())
        #env_meta.chronics_handler.tell_id(pre_id)
        # globals()[dynamic_name].chronics_handler.tell_id(pre_id)
        #env.chronics_handler.tell_id(env.chronics_handler.real_data.get_id())
        # self.id += 1
        #print(self.env.chronics_handler.real_data.get_id())

        # new_obs = env_meta.reset()
        # print(env_meta.chronics_handler.real_data.get_id())
        # new_obs = globals()[dynamic_name].reset()
        # print(globals()[dynamic_name].chronics_handler.real_data.get_id())

        mix_names = list(self.env.keys())
        #print(mix_names)
        mix_id = [1,2,3,4,0]
        
        self.env[mix_names[mix_id[self.num]]].set_id(self.id)
        new_obs = self.env[mix_names[mix_id[self.num]]].reset()
        print(self.id)
        print(self.num)
        print(self.env[mix_names[mix_id[self.num]]].chronics_handler.real_data.get_id())
        
        # mix_names = list(self.env.keys())
        # self.env[mix_names[0]].set_id(1)
        # #print(env.chronics_handler.real_data.get_id())
        # new_obs = self.env[mix_names[0]].reset()
        # print(self.env[mix_names[0]].chronics_handler.real_data.get_id())

        state = self.convert_obs(new_obs)

        alive_steps = 0
        states = []  # buffer
        actions = []
        new_states = []
        dones = []
        rewards = []
        actions_probs = []

        while (alive_steps < self._training_param.max_step):
            a, a_prob = self.A2C_worker.predict_movement(state)
            act = self.convert_act(a)
            # new_obs, reward, done, info = env_meta.step(act)
            #new_obs, reward, done, info = globals()[dynamic_name].step(act)
            new_obs, reward, done, info = self.env[mix_names[mix_id[self.num]]].step(act)
            
            new_state = self.convert_obs(new_obs)

            # 记录一条轨迹
            states.append(state)
            actions.append(a)
            rewards.append(reward)
            dones.append(done)
            new_states.append(new_state)
            #alive_steps += 1

            state = new_state

            if done or alive_steps == 2000:
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
                    loss_entropy = tf.reduce_mean(loss_entropy) * self.training_param.entropy_coeff
                    loss_actor += loss_entropy
                grads_a = tape.gradient(loss_actor, self.A2C_worker.model_policy_head.trainable_variables)

                # for g, w in zip(grads_a, meta_a_w):
                #     w -= 1e-4 * g / N_TASK

                with tf.GradientTape() as tape:
                    # critic loss
                    value_pred = self.A2C_worker.model_critic_head(states)
                    loss_critic = loss_func(value_pred, V_target)

                grads_c = tape.gradient(loss_critic, self.A2C_worker.model_critic_head.trainable_variables)
                # for g, w in zip(grads_c, meta_c_w):
                #     w -= 1e-4 * g / N_TASK

                # grads_a = self.normalize_gradients(grads_a)
                # grads_c = self.normalize_gradients(grads_c)

                # if alive_steps < 1000:
                #     grads_a = [grad * 0.6 for grad in grads_a]
                #     grads_c = [grad * 0.6 for grad in grads_a]
                # else:
                #     grads_a = [grad * 0.3 for grad in grads_a]
                #     grads_c = [grad * 0.3 for grad in grads_c]

                # break
                # self.A2C.model_policy_head.set_weights(meta_a_w)
                # self.A2C.model_critic_head.set_weights(meta_c_w)
                meta_optimizer.apply_gradients(zip(grads_a, self.A2C.model_policy_head.trainable_variables))
                meta_optimizer.apply_gradients(zip(grads_c, self.A2C.model_critic_head.trainable_variables))
                print("meta end")
                
                break

            else:
                alive_steps += 1

        self.num += 1

        

    def normalize_gradients(self, grads):#适用于多元学习中不同任务的梯度贡献不均衡问题,将梯度调整到单位尺度
        # tf.reshape(g, [-1])：将每个梯度张量 g 展平为一个一维向量。
        # [tf.reshape(g, [-1]) for g in grads]：对每个梯度张量执行展平操作，并将所有展平后的梯度张量放入一个列表中。
        # tf.concat(..., axis=0)：将所有展平后的梯度张量沿着第0维（即行方向）拼接成一个大的向量。
        # tf.norm(...)：计算拼接后的大向量的L2范数（即所有元素的平方和的平方根）。
        norm = tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0))
        # 将每个梯度 g 除以计算得到的范数 norm，得到归一化后的梯度。为了避免除以0的情况，加了一个很小的数 1e-10 到范数中。
        return [g / (norm + 1e-10) for g in grads]

    def weighted_gradients(self, grads, weight):
        return [g * weight for g in grads]

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
            # mix_names = list(self.env.keys())
            # random_env = self.env[mix_names[_]]
            #
            # random_env.chronics_handler.tell_id(random.randint(0, 119))
            #
            # print("ID of chronic current folder:", random_env.chronics_handler.real_data.get_id())
            # # Reset the environment
            # #env.reset()
            # random_env.set_id(random_env.chronics_handler.real_data.get_id())
            # random_env.reset()
            # tasks.append((mix_names[_], random_env.chronics_handler.real_data.get_id()))
            #
            # del random_env
            # gc.collect()


            mix_names = list(self.env.keys())
            _ = N_task-1-_

            self.env[mix_names[_]].chronics_handler.tell_id(0)
            self.env[mix_names[_]].reset()
            print("ID of chronic current folder:", self.env[mix_names[_]].chronics_handler.real_data.get_id())
            # Reset the environment
            # env.reset()
            self.env[mix_names[_]].set_id(0)
            self.env[mix_names[_]].reset()
            print("ID of chronic current folder:", self.env[mix_names[_]].chronics_handler.real_data.get_id())
            tasks.append((mix_names[_], self.env[mix_names[_]].chronics_handler.real_data.get_id()))

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
        step = 0
        self.init_training()
        #self.env_list.append(env.chronics_handler.real_data.get_id())

        with tqdm(total=train_step, disable=False, miniters=1, mininterval=3) as pbar:
            train_summary_writer = tf.summary.create_file_writer(logdir)
            # meta_a_w = self.A2C.model_policy_head.get_weights()
            # meta_c_w = self.A2C.model_critic_head.get_weights()

            # N_task = 5
            # env_list = self.random_task(N_task)

            while(step < train_step):
                meta_a_w = self.A2C.model_policy_head.get_weights()
                meta_c_w = self.A2C.model_critic_head.get_weights()
                self.id += 1

                N_task = 5
                # env_list = self.random_task(N_task)
                weights_a = np.copy(meta_a_w)
                weights_c = np.copy(meta_c_w)

                # for env_name, _ in env_list:
                for _ in range(N_task):
                    print("env start")
                    done = False
                    # env_id = env[env_name]
                    # env_id.set_chunk_size(int(max(100, nb_ts_one_day)))
                    # env_id.set_id(_)
                    # env_id.reset()
                    # print(env_id.chronics_handler.real_data.get_id())
                    # new_obs = env_id.reset()

                    # env[env_name].set_id(_)
                    # env[env_name].reset()
                    # print(env[env_name].chronics_handler.real_data.get_id())
                    # new_obs = env[env_name].reset()

                    
                    new_obs = env.reset()
                    print(env.chronics_handler.real_data.get_id())
                    self.env_list.append(env.chronics_handler.real_data.get_id())
                    state = self.convert_obs(new_obs)
                    self.A2C_worker.model_policy_head.set_weights(weights_a)
                    self.A2C_worker.model_critic_head.set_weights(weights_c)
                    while(alive_steps < self._training_param.max_step):
                        # if done:
                        #     #new_obs = env[env_name].reset()
                        #     new_obs = env.reset()
                        #     print(env.chronics_handler.real_data.get_id())
                        #     state = self.convert_obs(new_obs)
                        a, a_prob = self.A2C_worker.predict_movement(state)
                        act = self.convert_act(a)
                        #new_obs, reward, done, info = env[env_name].step(act)
                        new_obs, reward, done, info = env.step(act)
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

                        state = new_state

                        if done:
                            # Normalize rewards before training
                            # normalized_rewards = self.normalize_rewards(rewards)
                            # 回合更新
                            loss_actor, loss_critic = self.A2C_worker.train(np.array(states), np.array(actions), np.array(rewards),
                                                                        np.array(new_states), np.array(dones),
                                                                        np.array(actions_probs))
                            self.epoch_rewards.append(total_reward)
                            self.epoch_alive.append(alive_steps)
                            self.loss_actor_list.append(loss_actor)
                            self.loss_critic_list.append(loss_critic)
                            print(actions)

                            states.clear()  # buffer
                            actions.clear()
                            new_states.clear()
                            dones.clear()
                            rewards.clear()
                            actions_probs.clear()

                            # 进行元学习
                            # meta_a_w, meta_c_w = self.meta_lr(env_name, _, meta_a_w, meta_c_w, len(env_list))
                            self.meta_lr(env,N_task)

                            print("Survived [{}] steps".format(alive_steps))
                            print("Total reward [{}]".format(total_reward))

                            alive_steps = 0
                            total_reward = 0

                            break

                        else:
                            alive_steps += 1
                            if alive_steps == self._training_param.max_step:
                                # Normalize rewards before training
                                # normalized_rewards = self.normalize_rewards(rewards)
                                # 回合更新
                                loss_actor, loss_critic = self.A2C_worker.train(np.array(states), np.array(actions), np.array(rewards),
                                                                            np.array(new_states), np.array(dones),
                                                                            np.array(actions_probs))
                                self.epoch_rewards.append(total_reward)
                                self.epoch_alive.append(alive_steps)
                                self.loss_actor_list.append(loss_actor)
                                self.loss_critic_list.append(loss_critic)
                                #print(actions)

                                states.clear()  # buffer
                                actions.clear()
                                new_states.clear()
                                dones.clear()
                                rewards.clear()
                                actions_probs.clear()

                                # 进行元学习
                                # meta_a_w, meta_c_w = self.meta_lr(env_name, _, meta_a_w, meta_c_w, len(env_list))
                                self.meta_lr(env,N_task)
                                
                                print("Survived [{}] steps".format(alive_steps))
                                print("Total reward [{}]".format(total_reward))

                                alive_steps = 0
                                total_reward = 0

                                break

                        if step % 100 == 0 and len(self.epoch_rewards) >= 1:
                            with train_summary_writer.as_default():
                                mean_reward = np.mean(self.epoch_rewards)
                                mean_alive = np.mean(self.epoch_alive)
                                mean_loss_actor = np.mean(self.loss_actor_list)
                                mean_loss_critic = np.mean(self.loss_critic_list)

                                if len(self.epoch_rewards) >= 30:
                                    mean_reward_30 = np.mean(self.epoch_rewards[-30:])
                                    mean_alive_30 = np.mean(self.epoch_alive[-30:])
                                    mean_loss_actor = np.mean(self.loss_actor_list[-30:])
                                    mean_loss_critic = np.mean(self.loss_critic_list[-30:])

                                else:
                                    mean_reward_30 = mean_reward
                                    mean_alive_30 = mean_alive
                                    mean_loss_actor = mean_loss_actor
                                    mean_loss_critic = mean_loss_critic

                                if len(self.epoch_rewards) >= 1:
                                    tf.summary.scalar("mean_reward_30", mean_reward_30, step)
                                    tf.summary.scalar("mean_alive_30", mean_alive_30, step)
                                    tf.summary.scalar("mean_loss_actor", mean_loss_actor, step)
                                    tf.summary.scalar("mean_loss_critic", mean_loss_critic, step)



                        pbar.update(1)

                    print("env end")


                self.save(save_path)

        # 迭代结束
        self.save(save_path)