python -m torch.distributed.run --nproc_per_node=1 --master_port=12353 \
  test_batch.py \
  --protocol 'none' \
  --val_root "xxx" \
  --val_list "xxx/test_list.txt" \
  --arch swin_v2_b \
  --num_classes 2 \
  --input_size 224 \
  --batch_size 64 \
  --workers 4 \
  --resume 'xxx/full_swin_v2_base.pth' \
  --score_list 'xxx/result/full_swin_v2_base.txt'
