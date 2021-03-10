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
    parser.add_argument('--homo_src', action='store', type=str,
                        help='homography source path')
    parser.add_argument('--homo_dst', action='store', type=str,
                        help='homography destination path')
    parser.add_argument('--chessboard_h', type=int, default=7,
                        help='chessboard height')
    parser.add_argument('--chessboard_w', type=int, default=7,
                        help='chessboard width')
    parser.add_argument('--color_correct_src', action='store', type=str,
                        help='homography source path')
    parser.add_argument('--color_correct_dst', action='store', type=str,
                        help='homography destination path')
    parser.add_argument('--sharp_vid_pattern', action='store', type=str,
                        help='sharp video path pattern')
    parser.add_argument('--blur_vid_pattern', action='store', type=str,
                        help='blur video path pattern')
    parser.add_argument('--dest', action='store', type=str,
                        help='save path')

    args = parser.parse_args()

    roi = (args.crop_x, args.crop_y, args.crop_h, args.crop_w)
    if (not roi[0]) or (not roi[1]) or (not roi[2]) or (not roi[3]):
        roi = None
    datasetBuilder = DatasetBuilder(
            roi,
            (cv2.imread(args.homo_src), cv2.imread(args.homo_dst)),
            (args.chessboard_h, args.chessboard_w),
            (cv2.imread(args.color_correct_src),
                cv2.imread(args.color_correct_dst))
    )

    datasetBuilder.build(
            args.sharp_vid_pattern,
            args.blur_vid_pattern,
            args.dest
    )
