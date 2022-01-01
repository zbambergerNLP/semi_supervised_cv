image_dir = "./imagenette2-160/train"
csv_filename = "./imagenette2_160.csv"
image_dir_validation = "./imagenette2-160/val"
validation_filename = "./imagenette2_160_validation.txt"
SAVED_ENCODERS_DIR = './saved_encoders'
SAVED_FINE_TUNED_ENCODERS_DIR = './saved_fine_tuned_encoders'
NUM_OF_CLASSES = 10
HIDDEN_REPRESENTATION_DIM = 2048
MODEL_FILE_ENCODING = '.pt'
MODEL_CONFIGURATION_FILE_ENCODING = '.json'
RESNET_50 = 'resent50'
EPOCHS = 'epochs'
CUDA = 'cuda'
CPU = 'cpu'

# Configs
SEED = 'seed'
PRETRAINING_LEARNING_RATE = 'pretraining_learning_rate'
FINE_TUNING_LEARNING_RATE = 'fine_tuning_learning_rate'
PRETRAINING_EPOCHS = "pretraining_epochs"
FINE_TUNING_EPOCHS = 'fine_tuning_epochs'
PRETRAINING_MOMENTUM = 'pretraining_momentum'
FINE_TUNING_MOMENTUM = 'fine_tuning_momentum'
PRETRAINING_BATCH_SIZE = 'pretraining_batch_size'
FINE_TUNING_BATCH_SIZE = 'fine_tuning_batch_size'
PRETRAINING_M = "m"
MUL_FOR_NUM_KEYS = 'mul_for_num_of_keys'
ENCODER_OUTPUT_DIM = 'encoder_output_dim'
TEMPERATURE = "temperature"
PARAM_TRANSFER_MOMENTUM = 'param_transfer_momentum'


IMAGENETTE_LABEL_DICT = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute',
)

# METRICS
MINI_BATCH_INDEX = "mini_batch_index"
EPOCH_INDEX = "epoch_index"
MINI_BATCH_LOSS = "mini_batch_loss"
MINI_BATCH_ACCURACY = "mini_batch_accuracy"
EPOCH_LOSS = "epoch_loss"
EPOCH_ACCURACY = "epoch_accuracy"
