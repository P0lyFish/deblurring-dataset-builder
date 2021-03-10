python3 create_dataset.py \
    --crop_x 197 --crop_y 624 --crop_h 555 --crop_w 555 \
    --homo_src imgs/new/clock/sharp/00000050.png \
    --homo_dst imgs/new/clock/blur/00000050.png \
    --color_correct_src imgs/new/colorboard/sharp.png \
    --color_correct_dst imgs/new/colorboard/blur.png \
    --sharp_vid_pattern imgs/new/clock/sharp \
    --blur_vid_pattern imgs/new/clock/blur \
    --dest pilot
