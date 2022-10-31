python /home/manolotis/sandbox/robustness_benchmark/motionCNN/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/motionCNN/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/testing_no_past/" \
  --batch-size 64 \
  --n-jobs 32 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/robustness_benchmark/motionCNN/predictions/"\
  --model-name "xception71_no_past" \
  --model-name-addition "no_past_retrained"

python /home/manolotis/sandbox/robustness_benchmark/motionCNN/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/motionCNN/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/testing/" \
  --batch-size 64 \
  --n-jobs 32 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/robustness_benchmark/motionCNN/predictions/"\
  --model-name "xception71_no_past" \
  --model-name-addition "no_past_retrained_unperturbed"