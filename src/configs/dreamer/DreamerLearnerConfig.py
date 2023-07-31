from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        # self.MODEL_LR = 2e-4
        # self.ACTOR_LR = 5e-4
        # self.VALUE_LR = 5e-4
        # self.CAPACITY = 500000
        # self.MIN_BUFFER_SIZE = 100
        # self.MODEL_EPOCHS = 1
        # self.EPOCHS = 1
        # self.PPO_EPOCHS = 5
        # self.MODEL_BATCH_SIZE = 40
        # self.BATCH_SIZE = 40
        # self.SEQ_LENGTH = 50
        # self.N_SAMPLES = 1
        # self.TARGET_UPDATE = 1
        # self.DEVICE = 'cuda'
        # self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        # self.ENTROPY = 0.001
        # self.ENTROPY_ANNEALING = 0.99998
        # self.GRAD_CLIP_POLICY = 100.

        self.MODEL_LR = 5e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 250000 # transition总数
        self.MIN_BUFFER_SIZE = 500
        self.MODEL_EPOCHS = 60
        self.EPOCHS = 4
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 40

        self.max_MODEL_BATCH_SIZE = 2000
        self.holdout_ratio = 0.2
        self.mini_model_batch_size = 40

        self.BATCH_SIZE = 40
        self.SEQ_LENGTH = 20
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0

        self.HORIZON = 10
        self.rollout_min_length = 15
        self.rollout_max_length = 15
        self.rollout_min_step = 1e3
        self.rollout_max_step = 2e4

        self.use_MPCmodel = True

        self.rew = True
        self.rec = True
        self.avl = True
        self.pcont = True
        self.dis = True

        self.MPCHorizon = 6
        self.n_trajs = 4
        self.DeterPolForMo = False
        self.use_epsilon_MPC = False
        self.MPCepsilon = 0.02
        self.discount_MPC = False
        self.MPCgamma = 0.995
        self.m_r_predictor_epochs = 60
        self.m_r_predictor_batch_size = 40

        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.0

    def create_learner(self):
        return DreamerLearner(self)
