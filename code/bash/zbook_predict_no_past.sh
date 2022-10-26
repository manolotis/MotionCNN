python /home/manolotis/sandbox/robustness_benchmark/motionCNN/predict.py \
  --config /home/manolotis/sandbox/robustness_benchmark/motionCNN/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/testing_no_past" \
  --batch-size 8 \
  --n-jobs 2 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/robustness_benchmark/motionCNN/predictions/"\
  --model-name "xception71" \
  --model-name-addition "no_past"