#!/bin/bash
model_path="InternVL2_8B"
lmdeploy serve api_server --server-name 10.201.115.28 --server-port 8080 --backend turbomind --log-level INFO --model-name InternVL2_8B --cache-max-entry-count 0.9 --tp 1 ${model_path} --enable-prefix-caching --session-len 100000
