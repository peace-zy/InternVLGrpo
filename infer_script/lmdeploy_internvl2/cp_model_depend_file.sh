#!/bin/bash
dst_model=$1
echo 'copy file to '${dst_model}
cp InternVL2_8B/configuration_intern* ${dst_model}
cp InternVL2_8B/modeling_intern* ${dst_model}
cp InternVL2_8B/tokenization_internlm2_fast.py ${dst_model}
cp InternVL2_8B/conversation.py ${dst_model}
