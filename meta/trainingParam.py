import os
import json


class TrainingParam(object):
    _all_attr = ["SAVING_NUM", "gama", "lmbda", "entropy_coeff", "max_step", "loss_weight", "kwargs_archi", "list_attr_obs", "model_name"]

    def __init__(self,
                 SAVING_NUM=100,  # 每SAVING_NUM保存一次
                 gama=0.9,  # 折扣因子
                 lmbda=0.2,  # GAE的λ, 一般->0,远小于gama; λ=0, 为A=r+γV(s')-V(s), λ=1, 为A=sum(r)-V(s)
                 entropy_coeff=0.1,
                 max_step=8064, #设置智能体跑的最大步数
                 loss_weight=[1, 1],
                 kwargs_archi={},
                 list_attr_obs=[],  #状态空间特征名列表
                 model_name="NULL"
                 ):
        self.SAVING_NUM = SAVING_NUM
        self.gama = gama
        self.lmbda = lmbda
        self.entropy_coeff = entropy_coeff
        self.max_step = max_step
        self.loss_weight = loss_weight
        self.kwargs_archi = kwargs_archi
        self.list_attr_obs = list_attr_obs
        self.model_name = model_name

    def get_obs_size(self, env, list_attr_name):
        """get the size of the flatten observation"""
        res = 0
        for obs_attr_name in list_attr_name:
            beg_, end_, dtype_ = env.observation_space.get_indx_extract(obs_attr_name)
            res += end_ - beg_  # no "+1" needed because "end_" is exclude by python convention
        return res

    def to_dict(self):
        """serialize this instance to a dictionary."""
        res = {}
        for attr_nm in self._all_attr:
            tmp = getattr(self, attr_nm)
            res[attr_nm] = tmp if tmp is not None else None
        return res

    def save_as_json(self, path, name=None):
        """save this instance as a json"""
        res = self.to_dict()
        if name is None:
            name = "training_parameters.json"
        if not os.path.exists(path):
            raise RuntimeError(f'Directory "{path}" not found to save the training parameters')
        if not os.path.isdir(path):
            raise NotADirectoryError(f'"{path}" should be a directory')
        path_out = os.path.join(path, name)
        try:
            with open(path_out, "w", encoding="utf-8") as f:
                json.dump(res, fp=f, indent=4, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f'Failed to save json file: {e}')

    @classmethod
    def from_dict(cls, tmp):
        """initialize this instance from a dictionary"""
        if not isinstance(tmp, dict):
            raise RuntimeError(f'TrainingParam from dict must be called with a dictionary, and not {tmp}')
        res = cls()
        for attr_nm in cls._all_attr:
            setattr(res, attr_nm, tmp.get(attr_nm, None))
        return res

    @classmethod
    def from_json(cls, json_path):
        """initialize this instance from a json"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'No path are located at "{json_path}"')
        with open(json_path, "r", encoding="utf-8") as f:
            dict_ = json.load(f)
        return cls.from_dict(dict_)
