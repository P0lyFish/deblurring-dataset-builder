import glob
import os
import argparse
from tqdm import tqdm

SRC_VID_PATH_FORMAT = 'raw_datasets/{}/sharp/*.MP4'
DST_VID_PATH_FORMAT = 'raw_datasets/{}/blur/*.MP4'

SRC_SAVE_PATH_FORMAT = 'raw_datasets/{}/sharp/{:04d}'
DST_SAVE_PATH_FORMAT = 'raw_datasets/{}/blur/{:04d}'

# FFMPEG_EXTRACT_FRAMES = 'ffmpeg -i {} {}/%08d.png'
FFMPEG_EXTRACT_FRAMES = 'ffmpeg -i {} {}/%08d.png >/dev/null 2>&1'


def extract_frames(name):
    src_vid_paths = sorted(glob.glob(SRC_VID_PATH_FORMAT.format(name)))
    dst_vid_paths = sorted(glob.glob(DST_VID_PATH_FORMAT.format(name)))

    if len(src_vid_paths) != len(dst_vid_paths):
        raise ValueError('Number of sharp and blur videos are not equal!')

    process = tqdm(enumerate(zip(src_vid_paths, dst_vid_paths)), desc='Processing videos')
    for idx, (src_vid_path, dst_vid_path) in process:
        src_save_path = SRC_SAVE_PATH_FORMAT.format(name, idx)
        dst_save_path = DST_SAVE_PATH_FORMAT.format(name, idx)

        os.makedirs(src_save_path, exist_ok=True)
        os.makedirs(dst_save_path, exist_ok=True)

        os.system(FFMPEG_EXTRACT_FRAMES.format(src_vid_path, src_save_path))
        os.system(FFMPEG_EXTRACT_FRAMES.format(dst_vid_path, dst_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblurring dataset builder')
    parser.add_argument('--name', action='store', type=str,
                        help='dataset name')
    args = parser.parse_args()

    extract_frames(args.name)
