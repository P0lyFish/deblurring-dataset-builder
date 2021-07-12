import argparse
import cv2

from dataset_builder import DatasetBuilder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblurring dataset builder')
    parser.add_argument('--crop_x', action='store',
                        help='x coordinate of the cropped center',
                        type=int, default=None)
    parser.add_argument('--crop_y', action='store',
                        help='y coordinate of the cropped center',
                        type=int, default=None)
    parser.add_argument('--crop_h', action='store',
                        help='height of the cropped center',
                        type=int, default=None)
    parser.add_argument('--crop_w', action='store',
                        help='width of the cropped center',
                        type=int, default=None)
    parser.add_argument('--chessboard_h', type=int, default=7,
                        help='chessboard height')
    parser.add_argument('--chessboard_w', type=int, default=7,
                        help='chessboard width')
    parser.add_argument('--name', action='store', type=str,
                        help='dataset name')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--fps_ratio', action='store', default=1)

    args = parser.parse_args()

    roi = (args.crop_x, args.crop_y, args.crop_h, args.crop_w)
    if (not roi[0]) or (not roi[1]) or (not roi[2]) or (not roi[3]):
        roi = None

    DatasetBuilder(
            args.name,
            roi,
            (args.chessboard_h, args.chessboard_w),
            args.flip,
            args.fps_ratio
    ).build()

