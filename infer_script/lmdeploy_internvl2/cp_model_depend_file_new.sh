#!/bin/bash
dst_model=$1
echo 'copy file to '${dst_model}
root=Model/OpenGVLab/InternVL2-8B
cp ${root}/configuration_intern* ${dst_model}
cp ${root}/modeling_intern* ${dst_model}
cp ${root}/tokenization_internlm2_fast.py ${dst_model}
cp ${root}/conversation.py ${dst_model}
