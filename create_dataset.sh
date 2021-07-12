# python3.7 extract_frames.py --name checkerboards

python3.7 create_dataset.py \
    --name 100fps \
    --crop_x 223 --crop_y 648 --crop_h 512 --crop_w 512 \
    --flip \
    --fps_ratio 4
