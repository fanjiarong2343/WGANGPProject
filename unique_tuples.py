import os
from sequence.utils import *
from sequence.data_process import *
from config import *


if __name__ == '__main__':
    # 1. 找到4个generate文件
    output_dir = OUTPUT_DIR
    # output_dir = 'data/libpng'
    directories = os.listdir(output_dir)
    filter_dirs = []
    for bits in ['2bits', '4bits', '8bits', '16bits']:
        generate_list = list(filter(lambda x: x.find('generate') >= 0 and x.find(
            bits) >= 0, directories))
        generate_list.sort(reverse=True)  # list.sort(key=None, reverse=False) 当reverse=True时为降序排列，reverse=False为升序排列
        filter_dirs.append(generate_list[-1])
    print('filter_dirs:', filter_dirs)

    # 2. afl-cmin
    program = PROGRAM_PATH
    parameter = PARAMETER
    # program = 'data/libpng/pngfix'
    # parameter = '@@'
    for generate in filter_dirs:
        generate_dir = os.path.join(output_dir, generate)
        out_dir = 'cmin_out'
        cmd = "{} -i {} -o {} -- {} {}".format(AFLCMIN, generate_dir, out_dir, program, parameter)
        result, _ = subprocess_call(cmd)
        split = result[0].split()
        print('generate_dir:', generate_dir)
        # print('split:', split)

        index_tuples = split.index("Found") + 1
        print(int(split[index_tuples]))

        rm_cmd = "rm -rf {}".format(out_dir)
        _, _ = subprocess_call(rm_cmd)