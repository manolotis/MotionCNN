CUDA_VISIBLE_DEVICES=-1 python /home/manolotis/sandbox/robustness_benchmark/motionCNN/prerender.py \
    --data "/home/manolotis/sandbox/waymoMotion/data/reduced/tf_example/training/" \
    --out "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/training" \
    --n-shards 1
