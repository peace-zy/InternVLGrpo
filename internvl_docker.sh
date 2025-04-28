docker_name="internvl"
image_name=bj-harbor01.ke.com/aistudio/preset-images/huggingface:ubuntu-22.04-torch2.4.0-cuda12.4.1-24.09
sshd_port=8047
work_path=InternVL
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --privileged \
      --network=host \
      -d \
      -p 8801:8801 \
      -p 8802:8802 \
      -p 8803:8803 \
      -p 8804:8804 \
      -v /nfs/172.17.1.38/nvme4:/nfs/172.17.1.38/nvme4 \
      -v /nfs/172.17.1.38/nvme3:/nfs/172.17.1.38/nvme3 \
      -v /nfs/172.17.3.40:/nfs/172.17.3.40 \
      -v /chubao:/chubao \
      -v /mnt/:/mnt/ \
      -v /nvme4:/nvme4 \
      -w $work_path \
      --name $docker_name \
      $image_name \
      sleep infinity
      #[--sshd_port $sshd_port] --cmd 'sleep infinity'

