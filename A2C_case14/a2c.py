import os
import tensorflow as tf
from tqdm import tqdm
import warnings
import numpy as np
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct


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
        #self.action_space.filter_action(self._filter_action)
        print("转换后动作空间大小为{}".format(self.action_space.size()))

        self.env = env
        self._training_param = training_param
        #self.observation_space = observation_space
        self.observation_size = self._training_param.get_obs_size(env, self._training_param.list_attr_obs)
        print("挑选后状态空间大小为{}".format(self.observation_size))
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        #self.A2C = A2C(self.action_space.size(), self.observation_space.size(), lr=self.lr, training_param=self._training_param)
        self.A2C = A2C(self.action_space.size(), self.observation_size, lr_actor = self.lr_actor,
                       lr_critic = self.lr_critic, training_param=self._training_param)


    def _filter_action(self, action):
        MAX_ELEM = 4
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def init_training(self):
        self.epoch_rewards = []  # 存储平均每条轨迹reward
        self.epoch_alive = []  # 存储平均每条存活步数
        self.loss_actor_list = []  # 存储每轮loss
        self.loss_critic_list = []
        self.A2C = A2C(self.action_space.size(), self.observation_size, lr_actor = self.lr_actor,
                       lr_critic = self.lr_critic, training_param=self._training_param)

    def convert_obs(self, observation):
        # Made a custom version to normalize per attribute
        # return observation.to_vect()
        li_vect = []
        #for el in observation.attr_list_vect:
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

    def normalize_rewards(self, rewards):
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-10  # Prevent division by zero
        normalized_rewards = (rewards - mean_reward) / std_reward
        return normalized_rewards

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
        done = False
        self.init_training()
        new_obs = env.reset()
        state = self.convert_obs(new_obs)

        with tqdm(total=train_step, disable=False, miniters=1, mininterval=3) as pbar:
            train_summary_writer = tf.summary.create_file_writer(logdir)
            while(step < train_step):
                while(alive_steps <= self._training_param.max_step):
                    if done:
                        new_obs = env.reset()
                        state = self.convert_obs(new_obs)

                    a, a_prob = self.A2C.predict_movement(state)
                    act = self.convert_act(a)
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

                    if done or alive_steps == self._training_param.max_step:
                        # Normalize rewards before training
                        #normalized_rewards = self.normalize_rewards(rewards)
                        # 回合更新
                        loss_actor, loss_critic = self.A2C.train(np.array(states), np.array(actions), np.array(rewards),
                                                                    np.array(new_states), np.array(dones),
                                                                    np.array(actions_probs))
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


                        print("Survived [{}] steps".format(alive_steps))
                        print("Total reward [{}]".format(total_reward))

                        alive_steps = 0
                        total_reward = 0

                        break

                    else:
                        alive_steps += 1

                    if step % 100 == 0 and len(self.epoch_rewards) >= 1:
                        with train_summary_writer.as_default():
                            mean_reward = np.mean(self.epoch_rewards)
                            mean_alive = np.mean(self.epoch_alive)
                            mean_loss_actor = np.mean(self.loss_actor_list)
                            mean_loss_critic = np.mean(self.loss_critic_list)

                            if len(self.epoch_rewards) >= 30:
                                mean_reward_30 = np.mean(self.epoch_rewards[-30:])
                                mean_alive_30 = np.mean(self.epoch_alive[-30:])
                                if len(self.epoch_rewards) >= 50:
                                    mean_loss_actor = np.mean(self.loss_actor_list)
                                    mean_loss_critic = np.mean(self.loss_critic_list[-50:])

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

                self.save(save_path)

        # 迭代结束
        self.save(save_path)