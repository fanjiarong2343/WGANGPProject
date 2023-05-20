from sequence.utils import *
from config import *
from sequence.data_process import *


print("请核对FLAG是否正确：", FLAG)
s = input()
if s == 'Y':
    file_nparray()
else:
    exit()