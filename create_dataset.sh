python3 create_dataset.py \
    --crop_x 197 --crop_y 624 --crop_h 555 --crop_w 555 \
    --homo_src datasets/pilot/sharp/checkerboards/01.png \
    --homo_dst datasets/pilot/blur/checkerboards/01.png \
    --color_correct_src datasets/pilot/sharp/colorcards/01.png \
    --color_correct_dst datasets/pilot/blur/colorcards/01.png \
    --sharp_vid_pattern datasets/pilot/sharp/vids/001 \
    --blur_vid_pattern datasets/pilot/blur/vids/001 \
    --dest pilot \
    --flip
