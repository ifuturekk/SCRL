import torch


class Config(object):
    def __init__(self):
        self.data_folder = r'./data'
        self.data_path = r'./dataloader/data'
        self.exp_log_dir = 'experiment_logs_CDNet'  # HVAC: _tSNE
        # [f'comp{i}.csv' for i in range(2, 12)]
        self.file_names = ['comp1.csv', 'comp2.csv', 'comp3.csv', 'comp4.csv', 'comp5.csv', 'comp6.csv']
        self.faults = ['排气压力', '吸气压力']
        self.higher = [True, False]
        self.thresholds = [980, 220]
        self.time_len = 360
        self.prediction = 2
        self.step = 2

        # # HVAC
        # self.aim = 'train'  # 'HVAC'
        # # self.domains = [1, 2, 3, 4, 5, 6]
        # self.domains = ['new_1', 'new_2', 'new_3', 'new_4']
        # self.tasks = [([source for source in self.domains if source != target], target) for target in self.domains]
        # # self.tasks = [(['new_1'], 'new_2'), (['new_2'], 'new_3'), (['new_3'], 'new_4')]  # SIDGFD
        # self.in_channels = 8
        # self.classes = 3
        # self.batch_size = 1024

        # BGM: Bearing-4, Gear-3, Motor-3
        self.aim = 'Gear'
        # self.domains = [1, 2, 3]  # Raw data from Mo's paper
        self.domains = ['new_1', 'new_2', 'new_3']  # Randomly dropped
        # self.domains = ['new_1_scaled', 'new_2_scaled', 'new_3_scaled']  # Scaled after randomly dropped
        self.tasks = [([source for source in self.domains if source != target], target) for target in self.domains]
        # self.tasks = [([2, 3], 1)]
        self.in_channels = 1
        self.classes = 3
        self.batch_size = 128

        self.out_channels = 32
        self.dropout = 0.1
        self.feature_len = 32
        self.feature_matrix = 1  # Global Average Pooling (GAP)

        self.train_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.eval_device = torch.device('cpu')
        self.train_epoch = 100
        self.repeat_num = 10

        self.Conv1D = Conv1d()
        self.ResNet = ResNet()


class ResNet(object):
    def __init__(self):
        # raw: [64, 64, 128, 256, 512]
        # self.cfg = [4, 4, 8, 16, 32]
        self.cfg = [16, 16, 32]


class Conv1d(object):
    def __init__(self):
        self.kernel_size = 8
        self.stride = 2
        self.hidden_channels = [16, 32, 64]


