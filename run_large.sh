python3 run.py \
    --input_video ./assets/example_data/kubric/dynamic \
    --output_dir ./outputs \
    --encoder vitl \
    --input_size 518 \
    --max_res 1280 \
    --target_fps 24 \
    --seq_len 30 \
    --save_npz

# python3 run.py \
#     --input_video ./assets/example_videos/davis_rollercoaster.mp4 \
#     --output_dir ./outputs \
#     --encoder vitl \
#     --input_size 518 \
#     --max_res 1280 \
#     --target_fps -1 \
#     --save_npz