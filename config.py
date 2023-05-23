FLAG = 'LWCO'
# 'LWOC' 'LWCO' 'LWCA'
PROGRAM = 'size'
EXECUTE_DICT = {'exiv2': 'exiv2', 'djpeg': 'djpeg', 'jhead': 'jhead', 'readelf': 'readelf', 'libxml': 'xmllint', 'pngfix': 'pngfix', 'pngtest': 'pngtest', 'pngcp': 'pngcp', 'objdump': 'objdump', 'strip': 'strip', 'size': 'size'}
PARAMETER_DICT = {'exiv2': '@@', 'djpeg': '@@', 'jhead': '-v @@', 'readelf': '-a @@', 'libxml': '@@', 'pngfix': '@@', 'pngtest': '@@', 'pngcp': '@@ /dev/null', 'objdump': '-D @@', 'strip': '@@ -o /dev/null', 'size': '-A @@'}
EXECUTE = EXECUTE_DICT[PROGRAM]
PARAMETER = PARAMETER_DICT[PROGRAM]
PRECISION = '2'

MAX_COUNT = 7
C_DIM = 760


# MAX_EPOCH = 100 / 50
MAX_EPOCH = 80

# LWOC
STAGEI_G_LR = 1e-4
STAGEI_D_LR = 1.1e-4


# LWCO
# STAGEI_G_LR = 1e-4
# STAGEI_D_LR = 3e-4

OUTPUT_DIR = 'data/' + PROGRAM
PROGRAM_PATH = OUTPUT_DIR + '/' + EXECUTE
FORMAT_DIR = OUTPUT_DIR + '/format_set'
EDGE_DIR = OUTPUT_DIR + '/edge_set'
BITMAP_DIR = OUTPUT_DIR + '/bitmap_'
LOG_DIR = 'logs/' + PROGRAM
MODEL_PATH = 'model_weights/' + PROGRAM

SHOWMAP = '/home/fanjiarong/AFLplusplus/afl-showmap'
AFLCMIN = '/home/fanjiarong/AFLplusplus/afl-cmin'

SHUFFLE = True
BATCH_SIZE = 128
SPLIT_RATIO = 0.8
REFUSE_INFO = False
EDGES_INFO = False

MODEL_TYPE = 'conv1d'
if FLAG == 'LWOC':
    INDEX = 'format'
else:
    INDEX = 'edge'

Z_DIM = 128
EMBEDDING_DIM = 128

G_DIM = 32
D_DIM = 4

TRAIN_RATIO_I = 2
TRAIN_RATIO_II = 2

STAGEII_GI_LR = 1e-4
STAGEII_GII_LR = 1e-4
STAGEII_DII_LR = 7e-5

# 调节Discriminator的learning rate，一般要比Genenrator的小一点。
DECAY = 0.90
DECAY_EPOCHS = 10

OPT_TYPE = 'adam'
# 'rmsprop' 'adam' 'sgd'
