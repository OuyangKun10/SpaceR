#!/bin/bash

prefix="SpaceR-151k.part_"
total_parts=10

output_file="SpaceR-151k.tar.gz"  
target_directory="SpaceR-151k"  


mkdir -p "$target_directory"

echo "开始合并分卷文件..."
for ((i=0; i<total_parts; i++)); do
    formatted_i=$(printf "%02d" "$i")
    part_file="${prefix}${formatted_i}"
    if [ -f "$part_file" ]; then
        cat "$part_file" >> "$output_file"
    else
        echo "分卷文件 $part_file 不存在，合并中断。"
        exit 1
    fi
done

echo "合并完成，输出文件为：$output_file"

echo "开始解压文件到目录：$target_directory"
if [[ "$output_file" == *.tar.gz ]]; then
    tar -xzf "$output_file" -C "$target_directory"  
    if [[ $? -eq 0 ]]; then
        echo "解压完成。"
    else
        echo "解压失败，请检查文件格式是否正确。"
    fi
else
    echo "无法识别的文件类型，请检查合并后的文件扩展名。"
fi
