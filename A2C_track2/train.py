import time
from grid2op.MakeEnv import make
from grid2op.Action import *
from lightsim2grid.lightSimBackend import LightSimBackend

from grid2op.Reward import *
from trainingParam import TrainingParam
from a2c import Agent
from reward import NormalizedL2RPNReward

DEFAULT_NAME = "A2C"
def train(env,
        name=DEFAULT_NAME,
        train_step=1,
        save_path=None,
        load_path=None,
        logs_dir=None,
        training_param=None,
        verbose=True
        ):

    if training_param is None:
        training_param = TrainingParam()

    my_agent = Agent(env,
                    env.observation_space,
                    env.action_space,
                    lr_actor=5e-5,
                    lr_critic=1e-4,
                    training_param=training_param
                    )

    my_agent.train(env,
                train_step,
                save_path,
                logs_dir,
                training_param,
                verbose
                )


def main():
    backend = LightSimBackend()

    #env_name = "l2rpn_case14_sandbox"
    env_name = "l2rpn_neurips_2020_track2_small"

    env = make(env_name,
               #reward_class=L2RPNReward,
               reward_class=NormalizedL2RPNReward,
               action_class=TopologyChangeAndDispatchAction,
               backend=backend
            )
    print("最初状态空间大小为{}".format(env.observation_space.size()))
    print("最初动作空间大小为{}".format(env.action_space.size()))


    tp = TrainingParam()

    tp.SAVING_NUM = 1000
    #tp.gama = 0.99
    tp.gama = 0.90
    tp.max_step = 2000
    tp.loss_weight = []
    actor_weight = 0.5
    critic_weight = 1
    tp.loss_weight.append(actor_weight)
    tp.loss_weight.append(critic_weight)
    tp.model_name ="A2C-{}".format(int(time.time()))

    #case14所用特征
    # tp.list_attr_obs = ["prod_p", "prod_v", "load_p", "load_q",
    #                 "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
    #                 "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]
    tp.list_attr_obs = ["prod_p", "load_p",
                        "topo_vect", "rho",  "line_status",
                        "time_next_maintenance"]

    # tp.list_attr_obs = ["prod_p", "prod_q", "prod_v", "load_p", "load_q", "load_v", "actual_dispatch", "target_dispatch",
    #                     "time_before_cooldown_line", "time_before_cooldown_sub", "timestep_overflow", "gen_margin_up", "gen_margin_down",
    #                     "topo_vect", "rho", "line_status", "hour_of_day", "minute_of_hour"]

    # # case14所用网络
    # sizes = [800, 800, 494, 494]  # sizes of each hidden layers
    # PolicySize = [800, 576, 460]
    # CriticSize = [800, 512, 64]
    # tp.kwargs_archi = {'sharesizes': sizes,
    #                 'shareactivs': ["relu" for _ in sizes],  # all relu activation function
    #                 'Actorsizes':PolicySize,
    #                 'Actoractivs':["relu" for _ in PolicySize],
    #                 'Criticsizes': CriticSize,
    #                 'Criticactivs': ["relu" for _ in PolicySize]
    #                 }

    # sizes = [2048, 2048, 1536, 1536]
    # PolicySize = [2048, 1536, 1024, 2048]
    # CriticSize = [2048, 1016, 453, 64]

    # sizes = [4096, 4096, 2048, 2048, 1536, 1536]
    # PolicySize = [4096, 2048, 1536, 1024, 2048]
    # CriticSize = [4096, 2048, 1016, 453, 64]

    # sizes = [4096, 4096, 3056, 3056, 2048, 2048]
    # PolicySize = [4096, 3665, 2996, 2563, 1998]
    # CriticSize = [4096, 2048, 1016, 453, 64]

    #sizes = [8000, 8000, 8000, 8000, 8000, 4000, 2048, 2048]
    # sizes = [8000, 8000, 8000, 8000, 4000, 2048, 2048]
    # PolicySize = [4096, 3665, 2996, 2563, 1998]
    # CriticSize = [4096, 2048, 1016, 453, 64]

    # sizes = [8000, 8000, 4000, 4000]
    # PolicySize = [4096, 2996, 1998]
    # CriticSize = [4096, 2048, 453, 64]

    sizes = [3000, 3000, 900, 900]  # sizes of each hidden layers
    PolicySize = [3000, 2500, 2000]
    CriticSize = [3000, 1000, 64]

    tp.kwargs_archi = {'sharesizes': sizes,
                       'shareactivs': ["relu" for _ in sizes],  # all relu activation function
                       'Actorsizes': PolicySize,
                       'Actoractivs': ["relu" for _ in PolicySize],
                       'Criticsizes': CriticSize,
                       'Criticactivs': ["relu" for _ in CriticSize]
                       }

    save_path = "Outputs/Results/A2C/{}".format(tp.model_name)
    logs_dir = "Outputs/logs/A2C/{}".format(tp.model_name)
    load_path = None
    num_train_steps = 40000

    train(env,
        name=DEFAULT_NAME,
        train_step=num_train_steps,
        save_path=save_path,
        load_path=load_path,
        logs_dir=logs_dir,
        training_param=tp,
        verbose=False
        )

if __name__ == "__main__":
    main()