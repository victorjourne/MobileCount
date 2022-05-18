from easydict import EasyDict as edict

from datasets.Multiple.loader import CustomCCLabeler, CustomSHH, CustomWE, CustomJHU

# init
__C_DYN = edict()

cfg_data = __C_DYN

__C_DYN.IMAGE_SIZE = None
__C_DYN.TRAIN_SIZE = (576, 768)  # SHHB sizes (576, 768), WE raw sizes (576, 720) need padding, GCC (480, 848)
__C_DYN.LIST_C_DATASETS = [
    # (CustomGCC, '/workspace/data/GCC'),
    (CustomSHH, '/workspace/data/shanghaiTech/part_A_final/'),
    (CustomSHH, '/workspace/data/shanghaiTech/part_B_final/'),
    (CustomWE, '/workspace/data/worldExpo10_blurred'),
    (CustomCCLabeler, '/workspace/cclabeler/'),
    (CustomJHU, '/workspace/data/jhu_crowd_v2.0/'),
]

# __C_DYN.MEAN_STD = ([0.4355689, 0.41689757, 0.41106898], [0.27048737, 0.26903987, 0.28157565]) # SHHA+BKG
# __C_DYN.MEAN_STD = ([0.4551607, 0.45352426, 0.44445348], [0.24763596, 0.24388753, 0.25247648]) # SHHB+BKG
# __C_DYN.MEAN_STD = ([0.44195193, 0.42883006, 0.41919887], [0.25804275, 0.25455004, 0.26120335]) # SHHA+SHHB+BKG
# __C_DYN.MEAN_STD = ([0.43414703, 0.41423598, 0.40074137], [0.25675705, 0.25026417, 0.24924693]) # SHHA+SHHB
# __C_DYN.MEAN_STD = ([0.3220204, 0.31172827, 0.2942992], [0.23350126, 0.21823345, 0.19834155]) # SHHA+SHHB+GCC+BKG
# __C_DYN.MEAN_STD = ([0.4355689, 0.41689757, 0.41106898], [0.27048737, 0.26903987, 0.28157565]) # SHHA+BKG
# __C_DYN.MEAN_STD = ([0.48879814, 0.4907805, 0.4841541], [0.22630496, 0.22669446, 0.22931112]) # SHHA+SHHB+WE+BKG
__C_DYN.MEAN_STD = ([0.47024578, 0.45909372, 0.45179337], [0.24947935, 0.24922241, 0.25274596])  # SHHA+SHHB+WE+BKG+JHU
# __C_DYN.MEAN_STD = ([2.773511643408064, 2.826134968653417, 2.950241408655753], [4.145412486952323, 4.30860565292724, 4.555158225362852]) # SHHA+SHHB+WE+BKG+GCC
# Rappel
# SHHA
# __C_DYN.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
# SHHB
# __C_DYN.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
# GCC
# __C_DYN.MEAN_STD = ([0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.18484607339])
# BKG
# __C_DYN.MEAN_STD = ([0.45974895, 0.46210647, 0.46128437], [0.26007405, 0.26102796, 0.2821262])
# WE
# __C_DYN.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [0.22513884306, 0.225588873029, 0.22579960525])
# JHU
# __C_DYN.MEAN_STD = ([0.4337465, 0.39675272, 0.38812768], [0.28622445, 0.27810997, 0.28282893])

# __C_DYN.PROB = [0.2, 0.4, 0.4] # proba getting images
__C_DYN.COLLATE_FN = False
# better to remove because use in collate but no effect
# __C_DYN.LABEL_FACTOR = 1
__C_DYN.LOG_PARA = 2550.

# Negative value lead not to take in account those transforms
__C_DYN.RANDOM_DOWNOVER_SAMPLING = -1
__C_DYN.RANDOM_DOWN_SAMPLING = -1
__C_DYN.BRIGHTNESS_JITTER = 0.
__C_DYN.CONTRAST_JITTER = 0.
__C_DYN.SATURATION_JITTER = 0.
__C_DYN.HUE_JITTER = 0.

__C_DYN.RESUME_MODEL = '/data/models'
__C_DYN.TRAIN_BATCH_SIZE = 16
__C_DYN.VAL_BATCH_SIZE = 1
__C_DYN.PATH_SETTINGS = {
    'GCC__gt_folder': '/workspace/home/gameiroth/data/GCC/density/maps_adaptive_kernel/',
    'CC__index_filepath': '/workspace/cclabeler/users/background.json',
    'BG__gt_path': '/workspace/home/jourdanfa/data/density_maps/background/',
    'GD__gt_path': '/workspace/home/jourdanfa/data/density_maps/cclabeler/',
    'JHU__gt_path': '/workspace/home/jourdanfa/data/density_maps/jhu_crowd_v2.0/',
    'WE__dataset_weight': 1,
    'SHHA__dataset_weight': 1,
    'SHHB__dataset_weight': 1,
    'JHU__dataset_weight': 1,
    'BG__dataset_weight': 10,
}

# - GCC :
#    - GCC__gt_folder
#    - GCC__index_folder
#    - GCC__gt_format
# CC : 
#     required : CC__index_folder
#     optionnal : 
#        - BG__gt_format
#        - GD__gt_format
# - SHH :
#    - SHHA__gt_name_folder 
#    - SHHA__gt_format          
#    - SHHB__gt_name_folder
#    - SHHB__gt_format
# NOTE: ds gt folder must be in shh train / test folder (variable is the name of folder)
