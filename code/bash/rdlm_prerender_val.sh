CUDA_VISIBLE_DEVICES=-1 python /home/manolotis/sandbox/robustness_benchmark/motionCNN/prerender.py \
    --data "/home/manolotis/datasets/waymo/motion v1.0/uncompressed/tf_example/validation/" \
    --out "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/validation/" \
    --n-shards 1 \
    --n-jobs 48

