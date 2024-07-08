import numpy as np
from grid2op.Reward import L2RPNReward
from grid2op.dtypes import dt_float


class NormalizedL2RPNReward(L2RPNReward):
    def __init__(self, logger=None):
        super().__init__(logger=logger)

    def initialize(self, env):
        super().initialize(env)
        self.reward_min = dt_float(-1.0)
        #self.reward_max = dt_float(env.backend.n_line)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        #if not is_done and not has_error and not is_illegal and not is_ambiguous:
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = line_cap.sum()/env.backend.n_line
        else:
            res = self.reward_min

        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - x ** 2, np.zeros(x.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score
