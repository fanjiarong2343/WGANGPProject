#! /bin/bash

# 初始设置 cpu_id 为 1
cpu_id=1
time=86400


dirs=("djpeg")
declare -A map
map=(["djpeg"]="djpeg" ["exiv2"]="djpeg" ["jhead"]="djpeg" ["pngfix"]="pngfix" ["pngtest"]="pngfix" ["pngcp"]="pngfix")

declare -A target
target=(["djpeg"]="./djpeg @@" ["exiv2"]="./exiv2 @@" ["jhead"]="./jhead -v @@" ["pngfix"]="./pngfix @@" ["pngtest"]="./pngtest @@" ["pngcp"]="./pngcp @@ /dev/null" ["readelf"]="./readelf -a @@" ["libxml"]="./xmllint @@")

exp1_cmin=('cmin_1_random' 'cmin_1_format' 'cmin_1_generated' 'cmin_1_formatgen')
exp2_cmin=('cmin_2_origcminal' 'cmin_2_concat' 'cmin_2_gauss' 'cmin_2_generated' 'cmin_2_formatgen')

exp1_output=('out_1_random' 'out_1_format' 'out_1_generated' 'out_1_formatgen')
exp2_output=('out_2_original' 'out_2_concat' 'out_2_gauss' 'out_2_generated' 'out_2_formatgen')


for dir in ${dirs[@]}
do
    # 文件夹检查
    if [ ! -d data/$dir ]
    then
        echo data/$dir 不是文件夹, 检查下一个
        continue
    fi

    echo 进入文件夹: data/$dir
    cd data/$dir

    # for in_dir in ${exp1_input[@]}
    # do
    #     if [ -d $in_dir ]  # 如果$in_dir文件夹存在，删除文件夹
    #     then
    #         rm -rf $in_dir
    #     fi
    # done
    # for in_dir in ${exp2_input[@]}
    # do
    #     if [ -d $in_dir ]  # 如果$in_dir文件夹存在，删除文件夹
    #     then
    #         rm -rf $in_dir
    #     fi
    # done

    if [[ $dir -eq 'djpeg' || $dir -eq 'pngfix' || $dir -eq 'readelf' ]]
    then
        mkdir in_1_format
        id=0
        for file in `ls format_set` ; do
            cp format_set/$file in_1_format/$file
            # 更新id
            id=`expr $id + 1`
            if [[ $id -eq 1280 ]]; then
                break
            fi
        done
        in_1_format=in_1_format
        generated_2bits=$(ls | grep generated | grep 2bits | head -n 1)
        edges_generated_2bits=$(ls | grep generated | grep 2bits | tail -n 1)
    else
        in_1_format=../${map[$dir]}/in_1_format
        generated_2bits=../${map[$dir]}/$(ls ../${map[$dir]} | grep generated | grep 2bits | head -n 1)
        edges_generated_2bits=$(ls | grep generated | grep 2bits | tail -n 1)     
    fi

    # pwd='/home/fanjiarong/文档/WGANGPProject'
    # ln -s $pwd/data/$dir/$edges_generated_2bits/all/original in_2_original
    # ln -s $pwd/data/$dir/$edges_generated_2bits/all/concat in_2_concat
    # ln -s $pwd/data/$dir/$edges_generated_2bits/all/random in_2_gauss
    in_1_random=../../random
    in_1_generated=$generated_2bits
    in_2_original=$edges_generated_2bits/all/original
    in_2_concat=$edges_generated_2bits/all/concat
    in_2_gauss=$edges_generated_2bits/all/random

    mkdir in_1_formatgen
    mkdir in_2_generated
    mkdir in_2_formatgen

    # in_1_formatgen
    id=0
    for file in `ls $in_1_format` ; do
        cp $in_1_format/$file in_1_formatgen/$file
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 640 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_1_generated` ; do
        cp $in_1_generated/$file in_1_formatgen/$file
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 640 ]]; then
            break
        fi
    done

    # in_2_generated
    id=0
    for file in `ls $in_2_original` ; do
        cp $in_2_original/$file in_2_generated/$file:0
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 426 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_2_concat` ; do
        cp $in_2_concat/$file in_2_generated/$file:1
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 426 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_2_gauss` ; do
        cp $in_2_gauss/$file in_2_generated/$file:2
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 428 ]]; then
            break
        fi
    done
    
    # in_2_formatgen
    id=0
    for file in `ls $in_1_format` ; do
        cp $in_1_format/$file in_2_formatgen/$file
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 320 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_2_original` ; do
        cp $in_2_original/$file in_2_formatgen/$file:0
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 320 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_2_concat` ; do
        cp $in_2_concat/$file in_2_formatgen/$file:1
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 320 ]]; then
            break
        fi
    done
    id=0
    for file in `ls $in_2_gauss` ; do
        cp $in_2_gauss/$file in_2_formatgen/$file:2
        # 更新id
        id=`expr $id + 1`
        if [[ $id -eq 320 ]]; then
            break
        fi
    done

    exp1_input=($in_1_random $in_1_format $in_1_generated 'in_1_formatgen')
    exp2_input=($in_2_original $in_2_concat $in_2_gauss 'in_2_generated' 'in_2_formatgen')

    # 核对数量
    for in_dir in ${exp1_input[@]}
    do
        echo $in_dir 文件夹
        num=$(ls $in_dir | wc -l)
        if [ $num -ne 1280 ]
        then
            echo $in_dir 文件夹包含的测试用例数为 $num !!!
        fi
    done
    for in_dir in ${exp2_input[@]}
    do
        echo $in_dir 文件夹
        num=$(ls $in_dir | wc -l)
        if [ $num -ne 1280 ]
        then
            echo $in_dir 文件夹包含的测试用例数为 $num !!!
        fi
    done

    # # afl-cmin
    # num_1=${#exp1_input[@]}
    # num_2=${#exp2_input[@]}
    # for(( i=0;i<$num_1;i++ ))
    # do
    #     /home/fanjiarong/AFLplusplus/afl-cmin -i ${exp1_input[i]} -o ${exp1_cmin[i]} -- ${target[$dir]}
    # done
    # for(( i=0;i<$num_2;i++ ))
    # do
    #     /home/fanjiarong/AFLplusplus/afl-cmin -i ${exp2_input[i]} -o ${exp2_cmin[i]} -- ${target[$dir]}
    # done 

    # # afl-fuzz
    # for(( i=0;i<$num_1;i++ ))
    # do
    #     /home/fanjiarong/AFLplusplus/afl-fuzz -b $cpu_id -m none -t 5000 -i ${exp1_cmin[i]} -o ${exp1_output[i]} -V $time -- ${target[$dir]} &
    #     # 更新cpu_id
    #     cpu_id=`expr $cpu_id + 1`
    # done
    # for(( i=0;i<$num_2;i++ ))
    # do
    #     /home/fanjiarong/AFLplusplus/afl-fuzz -b $cpu_id -m none -t 5000 -i ${exp2_cmin[i]} -o ${exp2_output[i]} -V $time -- ${target[$dir]} & 
    #     # 更新cpu_id
    #     cpu_id=`expr $cpu_id + 1`
    # done 

    cd ..
done