import six
from ast import literal_eval
import codecs
import yaml

# 使用的时候如果直接赋值出去，默认是不可变的，如果需要再赋值一定注意
class PjConfig(dict):
    def __init__(self, *args, **kwargs):
        super(PjConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = PjConfig()
        return self[key]

    def __setitem__(self, key, value):
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but PjConfig is immutable'.format(
                    key, value
                )
            )
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(PjConfig, self).__setitem__(key, value)

    def update_from_Config(self, other):
        if isinstance(other, dict):
            other = PjConfig(other)
        assert isinstance(other, PjConfig)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError("Non-existent config key: {}".format(key))
        self.check()

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".format(
                    config_list
                )
            )
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError("Non-existent config key: {}".format(key))
        self.check()

    def update_from_file(self, config_file):
        with codecs.open(config_file, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        self.update_from_Config(dic)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, PjConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable

    def check(self):
        if cfg.PREP.THICKNESS % 2 != 1:
            raise ValueError("2.5D预处理厚度 {} 不是奇数".format(cfg.TRAIN.THICKNESS))


cfg = PjConfig()

"""数据集配置"""
# 数据集名称
cfg.DATA.NAME = "lits"
# 输入的2D或3D图像路径
cfg.DATA.INPUTS_PATH = ".image"
# 预处理输出npz路径
cfg.DATA.PREP_PATH = ".image"
# z 方向初始化可以指定一个独立的输出文件路径
cfg.DATA.Z_PREP_PATH = cfg.DATA.PREP_PATH

""" 预处理配置 """
# 预处理进行的平面
cfg.PREP.PLANE = "xy"
# 处理过程中所有比这个数字大的标签都设为前景
cfg.PREP.FRONT = 1
# 是否将数据只 crop 到前景
cfg.PREP.CROP = False
# 是否对数据插值改变大小
cfg.PREP.INTERP = False
# 进行插值的话目标片间间隔是多少，单位mm，-1的维度不会进行插值
cfg.PREP.INTERP_PIXDIM = (-1, -1, 1.0)
# 是否进行窗口化，在预处理阶段不建议做，灵活性太低
cfg.PREP.WINDOW = False
# 窗宽窗位
cfg.PREP.WWWC = (400, 0)
# 丢弃前景数量少于thresh的slice
cfg.PREP.THRESH = 256
# 3D的数据在开始切割之前pad到这个大小，-1的维度会放着不动
cfg.PREP.SIZE = (512, 512, -1)
# 2.5D预处理一片的厚度
cfg.PREP.THICKNESS = 3
# 预处理过程中多少组数据组成一个npz文件
# 可以先跑bs=1，看看一对数据多大；尽量至少将训练数据分入10个npz，否则分训练和验证集的时候会很不准
# 这个值不建议给成 2^n，这样更利于随机打乱数据
cfg.PREP.BATCH_SIZE = 4
