import math
import os
import re
import subprocess
from typing import List, Optional, Tuple, Union
import numpy as np


def subprocess_call(cmd: str, encoding="utf-8"):
    """
    开一个子进程执行命令, 并返回结果, 命令和返回结果的编码默认为utf-8编码.

    Args:
        cmd (str): 命令内容
        encoding (str): 编码格式
    Returns:
        Tuple[str, str]: (stdout, stderr)
    """
    try:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             encoding=encoding)
        result = p.communicate()
        p.terminate()
        returncode = p.returncode
        return result, returncode
    except Exception as e:
        print(e)
    return None, -1


def get_max_file_bytes(dir_name: str, recursive: bool=False)-> int:
    """
    获取一个目录下所有文件80%的最大字节数
    Args:
        dir_name (str): 文件夹的路径 data/djpeg/training_set
        recursive (bool): 是否递归子文件夹
    Returns:
        int: 80%的最大字节数
    """
    cwd = os.getcwd()
    need_recursive = "R" if recursive else ""
    cmd = "ls -S{} {}/{}/".format(need_recursive, cwd, dir_name)

    result, _ = subprocess_call(cmd)
    split = result[0].split()

    # if there exists no files
    if len(split) == 0:
        return -1
    
    file_name = split[int(0.2 * len(split))].rstrip()  # 删除字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符
    file_path = "{}/{}/{}".format(cwd, dir_name, file_name)
    size = os.path.getsize(file_path)
    return size


def _1bits_np(bytes: bytearray,
              b: float = 0.0, k: float = 1.0) -> np.ndarray:
    tmp = [j for j in bytes]
    arr = []
    for i in tmp:
        arr.extend([(i >> 7) & 0x01, (i >> 6) & 0x01, (i >> 5) & 0x01, (i >> 4) & 0x01, (i >> 3) & 0x01, (i >> 2) & 0x01, (i >> 1) & 0x01, i & 0x01])
    tmp = arr
    seed = np.array(tmp)
    seed = seed.astype('float32')
    seed = (seed - b) / k
    return seed


def _2bits_np(bytes: bytearray,
              b: float = 0.0, k: float = 3.0) -> np.ndarray:
    tmp = [j for j in bytes]
    arr = []
    for i in tmp:
        arr.extend([(i >> 6) & 0x03, (i >> 4) &
                   0x03, (i >> 2) & 0x03, i & 0x03])
    tmp = arr
    seed = np.array(tmp)
    seed = seed.astype('float32')
    seed = (seed - b) / k
    return seed


def _4bits_np(bytes: bytearray,
              b: float = 0.0, k: float = 15.0) -> np.ndarray:
    tmp = [j for j in bytes]
    arr = []
    for i in tmp:
        arr.extend([(i >> 4) & 0x0f, i & 0x0f])
    tmp = arr
    seed = np.array(tmp)
    seed = seed.astype('float32')
    seed = (seed - b) / k
    return seed


def _8bits_np(bytes: bytearray,
              b: float = 0.0, k: float = 255.0) -> np.ndarray:
    tmp = [j for j in bytes]
    seed = np.array(tmp)
    seed = seed.astype('float32')
    seed = (seed - b) / k
    return seed


def _16bits_np(bytes: bytearray,
               b: float = 0.0, k: float = 65535.0) -> np.ndarray:
    tmp = [j for j in bytes]
    seed = np.array(tmp)
    seed = seed.astype('uint8')

    ln: int = seed.shape[0]
    arr = [0 for i in range(ln//2)]
    arr = np.array(arr)
    for i in range(0, ln - 1, 2):
        arr[i // 2] = seed[i] * 256 + seed[i + 1]

    seed = arr.astype('float32')
    seed = (seed - b) / k
    return seed


# get vector representation of input
def vectorize_file(file_path: str, after_padding: int):
    """vectorize a file given in `file_path`, we will pad it until it reaches `after_padding`

    Args:
        file_path (str): file path
        after_padding (int): the length after padding

    Raises:
        ValueError: raised when len(file) > after_padding

    Returns:
        np.ndarray: the vectoried result
    """
    tmp = open(file_path, 'rb').read()
    ln = len(tmp)
    if ln >= after_padding:
        tmp = tmp[:after_padding]
    else:
        tmp = tmp + (after_padding - ln) * b'\x00'

    bytes = bytearray(tmp)
    _16bits_seed = _16bits_np(bytes)
    _8bits_seed = _8bits_np(bytes)
    _4bits_seed = _4bits_np(bytes)
    _2bits_seed = _2bits_np(bytes)
    _1bits_seed = _1bits_np(bytes)

    return _16bits_seed, _8bits_seed, _4bits_seed, _2bits_seed, _1bits_seed


def devectorize_1bits_np(arr: np.ndarray, b: float = 0.0, k: float = 1.0):
    # 将一个np向量，每个值在[0, 1]区间，映射为[0, 1]区间的字节向量 合并保存为二进制字节文件
    assert arr.shape.__len__() == 2, "测试用例数组的维度不为2, 形状应为(num_of_seeds, input_dim)"
    num_of_seeds: int = arr.shape[0]
    ln: int = arr.shape[1]

    extend_arr = np.round((arr + b) * k)  # 向上取整
    extend_arr = extend_arr.astype('uint8')

    tmp = np.zeros((num_of_seeds, ln // 8))
    for i in range(0, ln, 8):
        tmp[:, i // 8] = extend_arr[:, i] << 7 | extend_arr[:, i + 1] << 6 | extend_arr[:, i+2] << 5 | extend_arr[:, i + 3] << 4 | extend_arr[:, i+4] << 3 | extend_arr[:, i + 5] << 2 | extend_arr[:, i+6] << 1 | extend_arr[:, i + 7]
    extend_arr = np.array(tmp).astype('uint8')

    return num_of_seeds, extend_arr


def devectorize_2bits_np(arr: np.ndarray, b: float = 0.0, k: float = 3.0):
    # 将一个np向量，每个值在[0, 1]区间，映射为[0, 3]区间的字节向量 合并保存为二进制字节文件
    assert arr.shape.__len__() == 2, "测试用例数组的维度不为2, 形状应为(num_of_seeds, input_dim)"
    num_of_seeds: int = arr.shape[0]
    ln: int = arr.shape[1]

    extend_arr = np.round((arr + b) * k)  # 向上取整
    extend_arr = extend_arr.astype('uint8')

    tmp = np.zeros((num_of_seeds, ln // 4))
    for i in range(0, ln, 4):
        tmp[:, i // 4] = extend_arr[:, i] << 6 | extend_arr[:, i +
                                                            1] << 4 | extend_arr[:, i+2] << 2 | extend_arr[:, i + 3]
    extend_arr = np.array(tmp).astype('uint8')

    return num_of_seeds, extend_arr


def devectorize_4bits_np(arr: np.ndarray, b: float = 0.0, k: float = 15.0):
    # 将一个np向量，每个值在[0, 1]区间，映射为[0, 15]区间的字节向量 合并保存为二进制字节文件
    assert arr.shape.__len__() == 2, "测试用例数组的维度不为2, 形状应为(num_of_seeds, input_dim)"
    num_of_seeds: int = arr.shape[0]
    ln: int = arr.shape[1]

    extend_arr = np.round((arr + b) * k)  # 向上取整
    extend_arr = extend_arr.astype('uint8')

    tmp = np.zeros((num_of_seeds, ln // 2))
    for i in range(0, ln, 2):
        tmp[:, i // 2] = extend_arr[:, i] << 4 | extend_arr[:, i+1]
    extend_arr = np.array(tmp).astype('uint8')

    return num_of_seeds, extend_arr


def devectorize_8bits_np(arr: np.ndarray, b: float = 0.0, k: float = 255.0):
    # 将一个np向量，每个值在[0, 1]区间，映射为[0, 255]区间的字节向量
    assert arr.shape.__len__() == 2, "测试用例数组的维度不为2, 形状应为(num_of_seeds, input_dim)"
    num_of_seeds: int = arr.shape[0]

    extend_arr = np.round((arr + b) * k)  # 向上取整
    extend_arr = extend_arr.astype('uint8')

    return num_of_seeds, extend_arr


def devectorize_16bits_np(arr: np.ndarray, b: float = 0.0, k: float = 65535.0):
    '''
    https://www.runoob.com/numpy/numpy-dtype.html
    uint8 无符号整数(0 to 255)
    uint16 无符号整数(0 to 65535)
    uint32 无符号整数(0 to 4294967295)
    '''
    # 将一个np向量，每个值在[0, 1]区间，映射为[0, 65535]区间的字节向量
    assert arr.shape.__len__() == 2, "测试用例数组的维度不为2, 形状应为(num_of_seeds, input_dim)"
    num_of_seeds: int = arr.shape[0]

    extend_arr = np.round((arr + b) * k)  # 向上取整
    extend_arr = extend_arr.astype('uint16')
    return num_of_seeds, extend_arr


def save_vector(arr: Union[np.ndarray, bytearray], dir_name, dir_path, label='8bits'):
    # 将字节向量arr以二进制的方式储存到本地路径`dir_path/filename
    save_dir = os.path.join(dir_path, dir_name)
    print("save_dir:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将预处理np数组转化为字节下的np数组
    if label == '16bits':
        num_of_seeds, extend_arr = devectorize_16bits_np(arr)
    elif label == '8bits':
        num_of_seeds, extend_arr = devectorize_8bits_np(arr)
    elif label == '4bits':
        num_of_seeds, extend_arr = devectorize_4bits_np(arr)
    elif label == '2bits':
        num_of_seeds, extend_arr = devectorize_2bits_np(arr)
    elif label == '1bits':
        num_of_seeds, extend_arr = devectorize_1bits_np(arr)
    else:
        print("ERROR!")

    for i in range(num_of_seeds):
        filename = 'gen:id:' + str(i).zfill(4)
        savefile = os.path.join(save_dir, filename)
        with open(savefile, 'wb') as f:
            bytes_arr = bytearray(extend_arr[i].tobytes())
            f.write(bytes_arr)
        f.close()