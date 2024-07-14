import os
import numpy as np
from sqlalchemy import true
import tensorflow as tf
import time
from trainingParam import TrainingParam
from tensorflow.keras.initializers import HeNormal

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add, Reshape
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.losses import mean_squared_error,categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops


class A2C(object):
    """Constructs the desired actor critic network"""
    # 初始化方法，设置动作维度、观测维度、学习率以及训练参数，并调用构建网络方法

    # def __init__(self, action_size, observation_size, lr=1e-5,
    #             training_param=TrainingParam()):
    #     self.action_size = action_size
    #     self.observation_size = observation_size
    #     self.lr_ = lr
    #     self.training_param = training_param
    #     self.construct_network()
    def __init__(self, action_size, observation_size, lr_actor=1e-5, lr_critic=1e-4,
                 training_param=TrainingParam()):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.training_param = training_param
        self.construct_network()

    def custom_loss(self):
        def loss(data, y_pred):
            # 提取真实标签，即动作对应的one-hot编码
            y_true = data[:, :self.action_size]
            # 提取优势函数的值
            adv = data[:, self.action_size:]
            # 计算新的策略概率,每个元素表示相应位置的 one-hot 编码和预测概率的乘积
            newpolicy_probs = K.sum(y_true * y_pred, axis=1)
            # 计算策略损失
            loss_actor = -(adv * K.log(newpolicy_probs + 1e-10))
            loss_actor = tf.reduce_mean(loss_actor)
            # 计算熵损失，以增加探索性
            loss_entropy = -K.sum(y_pred * K.log(y_pred + 1e-10))
            loss_entropy = tf.reduce_mean(loss_entropy)
            # 将熵损失加到策略损失中
            loss_actor += loss_entropy * self.training_param.entropy_coeff
            # 返回最终的损失
            return loss_actor
        return loss

    def construct_network(self):
        initializer = HeNormal()
        #共享层
        input_layer = Input(shape=(self.observation_size,), name="observation")
        lay_share = input_layer
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["sharesizes"], self.training_param.kwargs_archi["shareactivs"])):
            lay_share = Dense(size, name="layer_shared_hidden{}".format(lay_num))(lay_share)  # put at self.action_size全连接层
            lay_share = Activation(act)(lay_share)  # 激活层

        # #actor网络
        # lay_actor = lay_share
        # for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Actorsizes"], self.training_param.kwargs_archi["Actoractivs"])):
        #     lay_actor = Dense(size, name="layer_actor_head_hidden{}".format(lay_num))(lay_actor)  # put at self.action_size全连接层
        #     lay_actor = Activation(act)(lay_actor)  # 激活层
        # soft_proba = Dense(self.action_size, name="layer_actor_head_output", activation="softmax",
        #                    kernel_initializer='uniform')(lay_actor)
        #
        # #critic网络
        # lay_critic = lay_share
        # for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Criticsizes"], self.training_param.kwargs_archi["Criticactivs"])):
        #     lay_critic = Dense(size, name="layer_critic_head_hidden{}".format(lay_num))(lay_critic)  # put at self.action_size全连接层
        #     lay_critic = Activation(act)(lay_critic)  # 激活层
        # v_output = Dense(1, name="layer_critic_head_output")(lay_critic)

        # actor网络
        lay_actor = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Actorsizes"],
                                                  self.training_param.kwargs_archi["Actoractivs"])):
            lay_actor = Dense(size, name="layer_actor_head_hidden{}".format(lay_num))(lay_actor)
            lay_actor = Activation(act)(lay_actor)
        soft_proba = Dense(self.action_size, name="layer_actor_head_output", activation="softmax",
                           kernel_initializer='uniform')(lay_actor)

        # critic网络
        lay_critic = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Criticsizes"],
                                                  self.training_param.kwargs_archi["Criticactivs"])):
            lay_critic = Dense(size, name="layer_critic_head_hidden{}".format(lay_num))(lay_critic)
            lay_critic = Activation(act)(lay_critic)
        v_output = Dense(1, name="layer_critic_head_output")(lay_critic)

        self.model_policy_head = Model(inputs=[input_layer], outputs=[soft_proba],name="Actor-{}".format(int(time.time())))
        self.model_critic_head = Model(inputs=[input_layer], outputs=[v_output],name="Critic-{}".format(int(time.time())))
        self.model             = Model(inputs=[input_layer], outputs=[soft_proba, v_output],name="A2C-{}".format(int(time.time())))

        # 分别编译actor和critic网络
        self.model_policy_head.compile(optimizer=Adam(learning_rate=self.lr_actor, clipnorm=0.5),
                                       loss=self.custom_loss())
        self.model_critic_head.compile(optimizer=Adam(learning_rate=self.lr_critic, clipnorm=0.1),
                                       loss=mean_squared_error)

        print(self.model.summary())

    @staticmethod
    @tf.function
    def silent_predict(model, data):
        return model(data, training=False)

    def predict_movement(self, data):
        #print("Before policy head prediction")
        #a_prob = self.model_policy_head.predict(data)  # 预测动作概率

        data = np.expand_dims(data, axis=0)  # 扩展数据维度
        a_prob = self.silent_predict(self.model_policy_head, data)  # 使用 tf.function 包装的预测函数
        #print("After policy head prediction")
        a_prob = a_prob.numpy()  # 将 EagerTensor 转换为 NumPy 数组

        opt_policy = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())  # 根据概率选择动作
        pa = np.squeeze(a_prob)  # 将预测的概率压缩成一维
        return int(opt_policy), pa  # 返回选择的动作和概率

    def get_advantages(self, states, dones, rewards, last_state): # 对每一回合计算所有步的优势函数 (adv), dones已经转换成done为0，非done为1
        #values = self.model_critic_head.predict(states) # 使用 Critic 网络预测所有状态的值
        values = self.silent_predict(self.model_critic_head, states)  # 使用 tf.function 包装的预测函数
        values = values.numpy()
        Advantage = [] # 初始化存储优势函数的列表
        gae = 0 # 初始化广义优势估计 (Generalized Advantage Estimation, GAE)
        #print("Before critic head prediction")
        #v = self.model_critic_head.predict(last_state) # 使用 Critic 网络预测最后一个状态的值
        v = self.silent_predict(self.model_critic_head, last_state)  # 使用 tf.function 包装的预测函数
        #print("After critic head prediction")
        v = v.numpy()
        values = np.append(values, v, axis=0) # 将最后一个状态的值追加到 values 中

        # 逆序遍历奖励，计算每一步的优势函数
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.training_param.gama * values[i + 1] * dones[i] - values[i] # 计算 temporal-difference (TD) 残差 delta,如果done, values[i + 1] * dones[i] 会为 0，从而忽略掉未来的状态值
            gae = delta + self.training_param.gama * self.training_param.lmbda * dones[i] * gae # 计算 GAE
            Advantage.insert(0, gae) # 将 GAE 插入 Advantage 列表的开头

        adv = np.array(Advantage) # 将优势函数列表转换为 numpy 数组
        adv_normalized = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # 将优势函数列表转换为 numpy 数组

        return adv_normalized

    def calculate_returns(self, rewards, dones):
        result = np.empty_like(rewards)  # 创建与奖励相同形状的空数组
        result[-1] = rewards[-1]  # 最后一个时间步的返回值等于最后一个奖励值
        # 逆序计算每个时间步的返回值
        for t in range(len(rewards) - 2, -1, -1): #从 len(rewards) - 2 开始逆序迭代到 -1 （包括 -1），步长为 -1
            result[t] = rewards[t] + self.training_param.gama * dones[t] * result[t + 1]
            # 当前时间步的返回值等于当前奖励加上折扣率乘以是否结束标志乘以下一时间步的返回值
        return result  # 返回计算好的所有时间步的返回值数组

    def train(self, s_batch, a_batch, r_batch, s2_batch, d_batch, pa_batch):
        """Trains networks to fit given parameters"""
        #求GAE
        donee = tf.Variable(d_batch, dtype=tf.float32) ## 将 d_batch 转换为 TensorFlow 变量
        done = tf.expand_dims((1.0 - donee), axis=1)  # done为0，非done为1，用于筛选非done,扩展 donee 的维度，将其转换为形状为 (batch_size, 1) 的二维张量
        last_state = np.expand_dims(s2_batch[-1], axis=0) ## 获取最后一个状态，扩展其维度以便于模型预测
        adv_batch = self.get_advantages(s_batch,done,r_batch,last_state) ## 计算优势函数
        self.advs = adv_batch

        # 求 V_target
        #V_last = self.model_critic_head.predict(last_state)  # 预测最后一个状态的价值
        V_last = self.silent_predict(self.model_critic_head, last_state)
        V_last = V_last.numpy()
        r_batch[-1] += self.training_param.gama * done[-1] * V_last  # 更新最后一个奖励值，加上折扣后的最后一个状态的价值
        V_target = self.calculate_returns(r_batch, done)  # 计算所有步的目标价值（回报）
        act_batch = tf.one_hot(a_batch, self.action_size)  # 将动作批次转换为 one-hot 编码
        #a_prob = self.model_policy_head.predict(s_batch)

        # # Debugging: print the targets and predictions
        # print("V_target: ", V_target)
        # print("Critic predictions: ", self.model_critic_head.predict(s_batch))

        # #编译
        # self.model.compile(optimizer=Adam(learning_rate=self.lr_, clipnorm=0.5),
        #                     loss=[self.custom_loss(), mean_squared_error],
        #                     loss_weights=self.training_param.loss_weight,  # 选critic_loss小的，entropy大的，adv正且大的，即actor_loss  的
        #                     run_eagerly=True)

        # # 定义callback类
        # class MyCallback(tf.keras.callbacks.Callback):
        #     def on_train_begin(self, logs={}):
        #         self.losses = []
        #         return
        #
        #     def on_batch_end(self, batch, logs={}):  # batch 为index, logs为当前batch的日志acc, loss...
        #         self.losses.append(logs.get('loss'))
        #         return
        # #训练：fit
        # cb = MyCallback()
        # h = self.model.fit(x=s_batch, y={'layer_actor_head_output':np.append(act_batch,adv_batch,axis = 1), 'layer_critic_head_output':V_target},
        #                     batch_size=32, epochs=3, shuffle=true,callbacks=[cb])

        # loss_actor = h.history["layer_actor_head_output_loss"]
        # loss_v = h.history["layer_critic_head_output_loss"]

        #return tf.reduce_mean(loss_actor), tf.reduce_mean(loss_v)

        #print("Before actor network training")
        actor_loss = self.model_policy_head.fit(x=s_batch, y=np.append(act_batch, adv_batch, axis=1), batch_size=32,
                                                epochs=3, shuffle=True, verbose=0)
        #print("After actor network training")

        #print("Before critic network training")
        critic_loss = self.model_critic_head.fit(x=s_batch, y=V_target, batch_size=32, epochs=10, shuffle=True,
                                                 verbose=0)
        #("After critic network training")

        return tf.reduce_mean(actor_loss.history["loss"]), tf.reduce_mean(critic_loss.history["loss"])

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_policy_model = "{}_policy_head".format(path_model)
        path_critic_model = "{}_critic_head".format(path_model)
        return path_model, path_policy_model, path_critic_model

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        path_model, path_policy_model, path_critic_model = self._get_path_model(path, name)
        self.model.save('{}.{}'.format(path_model, ext))
        self.model_policy_head.save('{}.{}'.format(path_policy_model, ext))
        self.model_critic_head.save('{}.{}'.format(path_critic_model, ext))
        print("Successfully saved network at: {}".format(path))

    def load_network(self, path, name=None, ext="h5"):
        path_model, path_policy_model, path_critic_model = self._get_path_model(path, name)
        self.model = load_model('{}.{}'.format(path_model, ext))
        self.model_policy_head = load_model('{}.{}'.format(path_policy_model, ext))
        self.model_critic_head = load_model('{}.{}'.format(path_critic_model, ext))
        print("Successfully loaded network from: {}".format(path))