import os
import random

# Configuration parameters
CURRENT_DIR = os.path.dirname(__file__)

# Object classification: Pointnet configuration parameters
class var_object_classification():
    NUM_OF_EPOCHS = 50       # Number of epochs to train
    NUM_OF_POINTS = 100      # Number of input points for the Pointnet input
    NUM_OF_DIMENSIONS = 3    # Number of dimensions for the Pointnet input
    INPUT_SHAPE = (NUM_OF_POINTS, NUM_OF_DIMENSIONS)
    NUM_OF_CATEGORIES = 101  # Number of categories for the Pointnet output      
    TEST_SIZE = 0.33         # Percentage used to split the data for the training and test set
    
    SEGMENT_RADIUS = 50     # Radius used to determine the size of the segments
    OBJECT_VOXEL_SIZE = 29   # Size of the voxel filter used to generate the seeds

# Scene classification: Pointnet configuration parameters
class var_scene_classification():
    NUM_OF_EPOCHS = 50       # Number of epochs to train
    NUM_OF_POINTS = 2048     # Number of input points for the Pointnet input
    NUM_OF_DIMENSIONS = 3    # Number of dimensions for the Pointnet input
    INPUT_SHAPE = (NUM_OF_POINTS, NUM_OF_DIMENSIONS)
    NUM_OF_FEATURES = 1024   # Size of the feature vector
    NUM_OF_CATEGORIES = 2    # Number of categories for the Pointnet output   
    TEST_SIZE = 0.33         # Percentage used to split the data for the training and test set
    NUM_OF_CANDIDATES = 3    # Number of candidates segments to be selected
    
    SEGMENT_RADIUS = 120.84  # Radius used to determine the size of the segments
    VOXEL_SIZE_SCENE = 0.01  # Size of the voxel filter used to reduce the amount of points
    VOXEL_SIZE_SEEDS = 50    # Size of the voxel filter used to generate the seeds

# Rotational subgroup voting: configuration parameters   
class var_RSV():
    TESALLATION_LEVEL = 60   # Number of votes per point
    SIGMA_t = 10             # Translation bandwidth
    SIGMA_R = 22.5             # Rotational bandwidth   

# RSVnet: configuration parameters   
class var_RSVnet():
    __v = var_scene_classification
    NUM_OF_EPOCHS = 20       # Number of epochs to train
    NUM_OF_FEATURES = __v.NUM_OF_FEATURES # Size of the feature vector
    NUM_OF_POINTS = 1024
    NUM_OF_CANDIDATES = 1   # Number of final candidates pose to be generated

# Paths to the different files of the repository
class file_paths():
    # Path to the linemod folder
    LINEMOD = os.path.join(CURRENT_DIR, "../Linemod")
    # Path to the folders containing the ground truth poses and the clouds
    LINEMOD_POSE = LINEMOD + '/pose/'
    LINEMOD_PLY = LINEMOD + '/ply/'
    # Path to save the data created by the ground truth generator 
    GTG_OBJECT_SEGMENTATION_DATA = LINEMOD + '/object_classification/data'
    GTG_OBJECT_SEEDS_R = LINEMOD + '/object_classification/GTG_OBJECT_SEEDS_R.hdf5'
    GTG_RSVNET_L = LINEMOD + '/RSVnet_local/data'
    GTG_RSVNET_G = LINEMOD + '/RSVnet_global/data'

    # Path to the PointNet folder
    POINTNET = os.path.join(CURRENT_DIR,'../PointNet')
    # Object classification 
    OBJ_CLASS = POINTNET + '/object_classification'
    OBJ_CLASS_POINTNET_LOAD = OBJ_CLASS + '/models/trained_model.h5'  # Path to the pre-trained model
    OBJ_CLASS_POINTNET_SAVE = OBJ_CLASS + '/models/new_model_saved.h5'
    OBJ_CLASS_DATA_TRAIN = OBJ_CLASS + '/data/train'
    OBJ_CLASS_DATA_TEST = OBJ_CLASS + '/data/test'

    # Scene classification 
    SCN_CLASS = POINTNET + '/scene_classification'
    SCN_CLASS_POINTNET_LOAD = SCN_CLASS + '/models/81_accuracy_model.h5'  # Path to the pre-trained model
    SCN_CLASS_POINTNET_SAVE = SCN_CLASS + '/models/new_model_saved.h5'  
    SCN_CLASS_DATA = SCN_CLASS + '/data'
    
    # Path to the RSVnet folder
    RSVNET_L = os.path.join(CURRENT_DIR, '../RSVnet_local')
    RSVNET_DATA_TRAIN_L = RSVNET_L + '/data/train'
    RSVNET_DATA_TEST_L = RSVNET_L + '/data/test'
    RSVNET_DATA_EVAL_L = RSVNET_L + '/data/eval'

    # Path to the pre-trained models
    RSVNET_PT_L = RSVNET_L + '/models/'
    
    # Path to the RSVnet_global folder
    RSVNET_G = os.path.join(CURRENT_DIR, '../RSVnet_global')
    RSVNET_DATA_TRAIN_G = RSVNET_G + '/data/train'
    RSVNET_DATA_TEST_G = RSVNET_G + '/data/test'
    RSVNET_DATA_EVAL_G = RSVNET_G + '/data/eval'
    
    # Path to the pre-trained models
    RSVNET_PT_G = RSVNET_G + '/models/'

    # Path to the point clouds
    OBJECT = os.path.join(CURRENT_DIR, '../clouds/objects/obj_000005.ply')
    OBJECT_SEEDS = os.path.join(CURRENT_DIR, '../clouds/objects/object_classification_seeds.ply') 
    SCENE = os.path.join(CURRENT_DIR, '/home/ant69/RSVnet/clouds/scenes/cropped.ply') #0150
    SCENE_SEGMENT = os.path.join(CURRENT_DIR, '../clouds/scenes/segment.ply')

    # Path to evaluation metrics data
    EVALUATION_METRICS_DATA = os.path.join(CURRENT_DIR, '../src/evaluation_metrics.json')

class Linemod_data():
    def __init__(self, object:str):
        self.object = object

    def getObjectSets(self):
        if self.object is 'can': 
            num_of_samples = 1195   
                   
            train_idx = [0,3,4,5,10,13,16,19,25,32,36,38,39,43,49,52,55,66,70,
                         71,75,78,99,116,119,124,130,134,144,154,156,159,161,
                         162,168,174,178,182,184,189,194,195,198,210,217,219,
                         225,241,257,258,265,289,296,366,379,385,392,407,419,
                         432,433,436,440,448,450,454,459,465,472,482,485,490,
                         496,497,504,522,527,547,550,558,596,600,624,629,633,
                         638,646,654,665,672,674,685,703,713,742,751,754,760,
                         768,776,780,788,794,801,804,808,813,824,827,829,838,
                         844,845,859,865,877,878,894,896,903,916,931,935,942,
                         954,962,983,1046,1054,1058,1060,1064,1065,1070,1077,
                         1081,1086,1089,1090,1093,1095,1096,1097,1098,1118,
                         1124,1128,1130,1154,1168,1179,1184,1187,1190]
            
            """ 50%
            train_idx = [0, 3, 4, 5, 10, 13, 16, 19, 25, 32, 36, 38, 39, 43, 
                         49, 52, 55, 66, 70, 71, 75, 78, 99, 116, 119, 124, 
                         130, 134, 144, 154, 156, 159, 161, 162, 168, 174, 178, 
                         182, 184, 189, 194, 195, 198, 210, 217, 219, 225, 241, 
                         257, 258, 265, 289, 296, 366, 379, 385, 392, 407, 419, 
                         432, 433, 436, 440, 448, 450, 454, 459, 465, 472, 482, 
                         485, 490, 496, 497, 504, 522, 527, 547, 550, 558, 596, 
                         600, 624, 629, 633, 638, 646, 654, 665, 672, 674, 685, 
                         703, 713, 742, 751, 754, 760, 768, 776, 780, 788, 794, 
                         801, 804, 808, 813, 824, 827, 829, 838, 844, 845, 859, 
                         865, 877, 878, 894, 896, 903, 916, 931, 935, 942, 954, 
                         962, 983, 1046, 1054, 1058, 1060, 1064, 1065, 1070, 
                         1077, 1081, 1086, 1089, 1090, 1093, 1095, 1096, 1097, 
                         1098, 1118, 1124, 1128, 1130, 1154, 1168, 1179, 1184, 
                         1187, 1190, 875, 731, 111, 356, 584, 867, 1000, 968, 
                         85, 160, 1185, 626, 201, 232, 230, 67, 995, 897, 216, 
                         932, 353, 886, 998, 323, 235, 1035, 891, 393, 150, 
                         303, 358, 384, 777, 140, 692, 1136, 365, 1066, 568, 
                         532, 1074, 396, 636, 645, 437, 1110, 466, 278, 1186, 
                         846, 88, 555, 53, 298, 103, 990, 802, 507, 758, 166, 
                         153, 869, 102, 291, 457, 186, 1100, 734, 283, 335, 72, 
                         516, 544, 857, 244, 579, 117, 313, 413, 811, 834, 31, 
                         513, 26, 429, 680, 899, 114, 324, 700, 169, 275, 761, 
                         281, 957, 121, 534, 566, 880, 956, 539, 418, 635, 357, 
                         628, 531, 882, 483, 664, 1008, 133, 1115, 1063, 54, 
                         336, 563, 453, 191, 595, 737, 1082, 607, 475, 1099, 
                         1027, 1034, 787, 409, 494, 530, 495, 1017, 421, 1149, 
                         349, 690, 369, 100, 552, 415, 445, 934, 1015, 123, 
                         463, 812, 789, 1001, 274, 572, 332, 677, 1028, 818, 
                         632, 213, 1018, 89, 397, 1031, 681, 1144, 286, 920, 
                         621, 726, 410, 944, 1029, 562, 651, 821, 333, 1013, 
                         951, 452, 746, 262, 520, 209, 687, 1053, 541, 467, 
                         889, 727, 250, 510, 1011, 967, 1026, 686, 237, 488, 
                         773, 294, 239, 673, 640, 1012, 370, 1024, 500, 770, 
                         1051, 644, 425, 879, 714, 434, 1148, 110, 938, 631, 
                         172, 347, 1045, 1067, 187, 1127, 469, 992, 831, 1071, 
                         902, 252, 414, 755, 48, 315, 222, 997, 9, 236, 696, 
                         985, 669, 519, 826, 288, 395, 1040, 881, 292, 1042, 
                         92, 199, 383, 972, 205, 382, 927, 167, 107, 1080, 911, 
                         1165, 740, 228, 374, 112, 976, 1133, 331, 1171, 711,
                          885, 1111, 756, 207, 247, 585, 135, 505, 481, 306, 
                          246, 866, 1166, 512, 423, 108, 314, 176, 609, 231, 
                          717, 1188, 427, 1010, 206, 643, 271, 44, 327, 775, 
                          479, 403, 994, 334, 779, 996, 1131, 449, 847, 1048, 
                          943, 613, 399, 255, 518, 446, 1156, 21, 973, 958, 
                          148, 724, 950, 1180, 402, 430, 732, 185, 625, 471, 
                          1085, 536, 386, 928, 214, 910, 698, 825, 538, 340, 
                          667, 693, 987, 1140, 697, 1092, 549, 424, 709, 603, 
                          799, 46, 221, 1159, 948, 657, 211, 659, 993, 243, 
                          854, 137, 1094, 542, 69, 523, 229, 83, 470, 1056, 
                          1030, 898, 716, 28, 682, 1084, 745, 535, 1161, 554, 
                          524, 966, 412, 460, 220, 215, 557, 81, 196, 1033, 
                          1123, 850, 781, 259, 193, 373, 151, 769, 805, 321, 
                          177, 533, 42, 656, 706, 253, 741, 573, 267, 233, 704, 
                          970, 122, 371, 766, 319, 223, 511, 1052, 354, 1055, 
                          295, 602, 971, 390, 926, 1155, 487, 612, 65, 571, 76, 
                          263, 593, 401, 551, 904, 582, 728, 1183, 873, 1103, 
                          1176, 905, 868, 783, 309, 784]
                """
        # Create a list with all the samples for the selected object  
        object_all = list()
        for i in range(num_of_samples): 
            object_all.append('{0:04d}'.format(i))
        # Create a list with the name of the samples used for training      
        train = list()
        for i in train_idx:
            train.append('{0:04d}'.format(i))
        # Get the list with the samples used for testing. We get it by 
        # substracting the two sets previously defined
        test = list(set(object_all) - set(train))
        test.sort(key = int) 
        #random.shuffle(test)

        return train, test
    
    def getTrainFilePaths(self):
        [train, _] = self.getObjectSets()
        [train_T, train_ply] = self.__getSetFilesPaths(train)
        return train_T, train_ply

    def getTestFilesPaths(self):
        [_, test] = self.getObjectSets()
        [test_T, test_ply] = self.__getSetFilesPaths(test)
        return test_T, test_ply

    def __getSetFilesPaths(self, set):
        
        file_paths_ply = list()
        file_paths_T = list()

        f = file_paths

        for it in set:
            fp_T  = f.LINEMOD_POSE +  it + '.txt'
            file_paths_T.append(fp_T)
            fp_ply  = f.LINEMOD_PLY +  it + '.ply'
            file_paths_ply.append(fp_ply)

        return file_paths_T, file_paths_ply
