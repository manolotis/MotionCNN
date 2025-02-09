MODEL_NAME=xception71
python /home/manolotis/sandbox/robustness_benchmark/motionCNN/train.py \
    --train-data "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/training" \
    --dev-data "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/validation" \
    --train-data-noisy "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/training_no_road" \
    --dev-data-noisy "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/validation_no_road" \
    --save /home/manolotis/sandbox/robustness_benchmark/motionCNN/trained_models/${MODEL_NAME}_no_road \
    --model ${MODEL_NAME} \
    --img-res 224 \
    --in-channels 25 \
    --time-limit 80 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 1 \
    --n-epochs 120 \
    --n-shards 16 \
    --patience 1