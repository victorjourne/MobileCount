from easydict import EasyDict as edict

# init
__C_JHU = edict()

cfg_data = __C_JHU

# __C_JHU.STD_SIZE = (576,720)
__C_JHU.TRAIN_SIZE = (576,768) # Init with SHHB size
__C_JHU.DATA_PATH = '/workspace/data/jhu_crowd_v2.0/'

__C_JHU.MEAN_STD = ([0.43424335, 0.39698952, 0.38906667], [0.28815442, 0.28000653, 0.2846415])

__C_JHU.LABEL_FACTOR = 1
__C_JHU.LOG_PARA = 2550.

__C_JHU.RESUME_MODEL = ''#model path
__C_JHU.TRAIN_BATCH_SIZE = 16 #imgs

__C_JHU.VAL_BATCH_SIZE = 1 #



