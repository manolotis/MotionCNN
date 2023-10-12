CUDA_VISIBLE_DEVICES=-1 python /home/manolotis/sandbox/robustness_benchmark/motionCNN/prerender.py \
    --data "/home/manolotis/sandbox/waymoMotion/data/reduced/tf_example/validation/" \
    --out "/home/manolotis/sandbox/robustness_benchmark/motionCNN/data/prerendered/testing/" \
    --use-vectorize \
    --n-shards 1
