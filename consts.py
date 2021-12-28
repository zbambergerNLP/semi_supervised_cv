image_dir = "./imagenette2-160/train"
csv_filename = "./imagenette2_160.csv"
image_dir_validation = "./imagenette2-160/val"
validation_filename =  "./imagenette2_160_validation.txt"
SAVED_ENCODERS_DIR = './saved_encoders'
NUM_OF_CLASSES = 10
HIDDEN_REPRESENTATION_DIM = 2048
SAVED_ENCODERS_DIR = './saved_encoders'
MODEL_FILE_ENCODING = '.pt'
MODEL_CONFIGURATION_FILE_ENCODING = '.json'
RESNET_50 = 'resent50'
EPOCHS = 'epochs'
CUDA = 'cude'
CPU = 'cpu'

# Configs
SEED = 'seed'
LEARNING_RATE = 'learning_rate'
MOMENTUM = 'momentum'
BATCH_SIZE = 'batch_size'
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