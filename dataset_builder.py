import cv2
import os
import os.path as osp
import glob
from tqdm import tqdm


from homography import Homography
from color_correction import MyColorCorrection
from utils import Cropper


HOM_SRC_FORMAT = 'raw_datasets/{}/calibration_data/checkerboard_sharp.png'
HOM_DST_FORMAT = 'raw_datasets/{}/calibration_data/checkerboard_blur.png'

COLOR_CORRECT_SRC_FORMAT = 'raw_datasets/{}/calibration_data/color_correction_sharp/**.png'
COLOR_CORRECT_DST_FORMAT = 'raw_datasets/{}/calibration_data/color_correction_blur/**.png'

SRC_VID_FORMAT = 'raw_datasets/{}/sharp/**/'
DST_VID_FORMAT = 'raw_datasets/{}/blur/**/'

SAVE_SRC_PATH_FORMAT = 'processed_datasets/{}/sharp/{:04d}/{:08d}.png'
SAVE_DST_PATH_FORMAT = 'processed_datasets/{}/blur/{:04d}/{:08d}.png'


class DatasetBuilder:
    def __init__(
            self,
            name,
            roi,
            chessboard_size,
            flip_src,
            fps_ratio
    ):
        self.name = name
        self.roi = roi

        # raw videos
        self.src_vid_paths = sorted(glob.glob(SRC_VID_FORMAT.format(name)))
        self.dst_vid_paths = sorted(glob.glob(DST_VID_FORMAT.format(name)))

        if len(self.src_vid_paths) != len(self.dst_vid_paths):
            raise ValueError('Number of sharp and blur videos are not equal!')
        print(f'{len(self.src_vid_paths)} videos found!')

        # Flip the images of the left camera
        self.flip_src = flip_src

        # Ratio of sharp fps and blur fps
        self.fps_ratio = fps_ratio

        # crop center
        self.cropper = Cropper(roi)

        # Read and cache all calibration data
        # homography data
        hom_src_path = HOM_SRC_FORMAT.format(name)
        hom_dst_path = HOM_DST_FORMAT.format(name)

        if (not osp.isfile(hom_src_path)) or (not osp.isfile(hom_dst_path)):
            raise ValueError('Missing homography data!')

        hom_src = cv2.imread(hom_src_path)
        hom_dst = cv2.imread(hom_dst_path)

        # color correction data
        cc_src_paths = sorted(glob.glob(COLOR_CORRECT_SRC_FORMAT.format(name)))
        cc_dst_paths = sorted(glob.glob(COLOR_CORRECT_DST_FORMAT.format(name)))

        if (len(cc_src_paths) != len(cc_dst_paths)) or (len(cc_src_paths) != len(self.src_vid_paths)):
            raise ValueError('Number of color correction frames is not valid!')

        self.ccs_src = [cv2.imread(cc_src_path) for cc_src_path in cc_src_paths]
        self.ccs_dst = [cv2.imread(cc_dst_path) for cc_dst_path in cc_dst_paths]

        # Sharp videos captured from beamsplitter-based system are flipped
        if self.flip:
            hom_src = self.flip(hom_src)
            self.ccs_src = [self.flip(ccs) for ccs in self.ccs_src]

        # Initialize calibration tool
        # Note:
        # homography: sharp -> blur
        # color correction: blur -> sharp
        self.warper = Homography(hom_src, hom_dst, chessboard_size)

        cv2.imwrite('debug1.png', self.crop(self.warp(self.ccs_src[0])))
        cv2.imwrite('debug2.png', self.crop(self.ccs_dst[0]))

        self.ccs_src = [self.warp(ccs) for ccs in self.ccs_src]

        # remove black margins of the camera
        self.ccs_src = [self.crop(ccs) for ccs in self.ccs_src]
        self.ccs_dst = [self.crop(ccs) for ccs in self.ccs_dst]

        self.color_transformer = [MyColorCorrection(cc_dst, cc_src) for cc_dst, cc_src in zip(self.ccs_dst, self.ccs_src)]

    def crop(self, img):
        return self.cropper(img)

    def warp(self, img):
        return self.warper(img)

    def correct_color(self, img, idx):
        return self.color_transformer[idx](img)

    def flip(self, img):
        if self.flip_src:
            img = cv2.flip(img, 1)
        return img

    def build_video(self, video_idx):
        src_vid_path = self.src_vid_paths[video_idx]
        dst_vid_path = self.dst_vid_paths[video_idx]

        src_frame_paths = sorted(glob.glob(osp.join(src_vid_path, '*.png')))
        dst_frame_paths = sorted(glob.glob(osp.join(dst_vid_path, '*.png')))

        # Process sharp video: crop, flip, and warp
        for frame_idx, src_frame_path in\
                tqdm(enumerate(src_frame_paths), total=len(src_frame_paths), desc='Processing sharp video'):
            src_frame = cv2.imread(src_frame_path)
            src_frame = self.flip(src_frame)

            src_frame_post = self.crop(self.warp(src_frame))

            save_src_path = SAVE_SRC_PATH_FORMAT.format(self.name, video_idx, frame_idx)
            os.makedirs(osp.dirname(save_src_path), exist_ok=True)
            cv2.imwrite(
                save_src_path,
                src_frame_post
            )

        # Process blur videos: crop and correct color
        for frame_idx, dst_frame_path in\
                tqdm(enumerate(dst_frame_paths), total=len(dst_frame_paths), desc='Processing blurry video'):
            dst_frame = cv2.imread(dst_frame_path)
            dst_frame_post = self.correct_color(self.crop(dst_frame), video_idx)

            save_dst_path = SAVE_DST_PATH_FORMAT.format(self.name, video_idx, frame_idx)
            os.makedirs(osp.dirname(save_dst_path), exist_ok=True)
            cv2.imwrite(
                save_dst_path,
                dst_frame_post
            )

    def build(self):
        for idx in range(len(self.src_vid_paths)):
            print(f'Processing video #{idx:04d}/{len(self.src_vid_paths):04d}')
            self.build_video(idx)
