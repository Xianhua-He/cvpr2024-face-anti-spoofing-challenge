python -m torch.distributed.run --nproc_per_node=1 --master_port=12353 \
  test_batch.py \
  --protocol 'p2.2' \
  --val_root "xxx/cvpr2024/data" \
  --val_list "xxx/cvpr2024/data/p2.2/dev_test.txt" \
  --arch resnet50 \
  --num_classes 2 \
  --input_size 224 \
  --batch_size 1024 \
  --workers 8 \
  --resume 'xxx/cvpr2024/submit/exp_p22/resnet50_epoch199_acc1_99.9762.pth' \
  --score_list 'xxx/cvpr2024/submit/result/exp_p22/exp_p22.txt'