#!/bin/sh

python gen_lane_result.py \
  --dataroot=exp_example \
  --testlist=exp_example/testlist.txt \
  --arch=resnet50_dilated_forseg_x8 \
  --net=Seg \
  --classes=5 \
  --fchannel=128 \
  --zoom_factor=8 \
  --workers=1 \
  --batch_size=2 \
  --print_freq=1 \
  --resume=exp_example/try_epoch_0.checkpoint.pth.tar \
  --savedir=exp_example/pngresult \
  --savedir_prob=exp_example/pngresult_prob