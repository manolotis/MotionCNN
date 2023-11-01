python /home/manolotis/sandbox/scenario_based_evaluation/motionCNN/predict.py \
  --config /home/manolotis/sandbox/scenario_based_evaluation/motionCNN/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/testing/" \
  --batch-size 64 \
  --n-jobs 32 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/scenario_based_evaluation/motionCNN/predictions/"\
  --model-name "xception71"