import cv2
import os
import os.path as osp
import glob


from homography import Homography
from color_correction import MyColorCorrection
from utils import Cropper


class DatasetBuilder:
    def __init__(
            self,
            roi,
            homography_refs,
            chessboard_size,
            color_correction_refs,
            flip_src
    ):
        # Flip the images of the left camera
        self.flip_src = flip_src

        # crop center
        self.cropper = Cropper(roi)

        # geometric alignment using homography
        hom_src, hom_dst = homography_refs
        hom_src = self.flip(hom_src)
        self.warper = Homography(hom_src, hom_dst, chessboard_size)

        # color correction
        cc_src, cc_dst = color_correction_refs
        cc_src = self.flip(cc_src)

        cc_src = self.warp(cc_src)
        cc_src, cc_dst = self.crop(cc_src), self.crop(cc_dst)

        self.color_transformer = MyColorCorrection(cc_dst, cc_src)

    def crop(self, img):
        return self.cropper(img)

    def warp(self, img):
        return self.warper(img)

    def correct_color(self, img):
        return self.color_transformer(img)

    def flip(self, img):
        if self.flip_src:
            img = cv2.flip(img, 1)
        return img

    def build_one_video(
            self,
            sharp_video_path,
            blur_video_path,
            dst,
            video_idx
    ):

        sharp_dst = osp.join(dst, '{:03d}/sharp'.format(video_idx))
        blur_dst = osp.join(dst, '{:03d}/blur'.format(video_idx))

        os.makedirs(osp.join(dst, '{:03d}'.format(video_idx)))
        os.makedirs(sharp_dst)
        os.makedirs(blur_dst)

        sharp_frame_paths = sorted(
            glob.glob(osp.join(sharp_video_path, '*.png'))
        )
        blur_frame_paths = sorted(
            glob.glob(osp.join(blur_video_path, '*.png'))
        )

        for frame_idx, (sharp_frame_path, blur_frame_path) in\
                enumerate(zip(sharp_frame_paths, blur_frame_paths)):
            sharp_frame = cv2.imread(sharp_frame_path)
            blur_frame = cv2.imread(blur_frame_path)
            sharp_frame = self.flip(sharp_frame)

            sharp_frame_post = self.crop(self.warp(sharp_frame))
            blur_frame_post = self.correct_color(self.crop(blur_frame))

            cv2.imwrite(
                osp.join(sharp_dst, '{:08d}.png'.format(frame_idx)),
                sharp_frame_post
            )

            cv2.imwrite(
                osp.join(blur_dst, '{:08d}.png'.format(frame_idx)),
                blur_frame_post
            )

    def build(
            self,
            sharp_video_path_pattern,
            blur_video_path_pattern,
            dst
    ):
        if osp.isdir(dst):
            raise ValueError('{} folder already exists'.format(dst))
        os.makedirs(dst)

        sharp_video_paths = sorted(glob.glob(sharp_video_path_pattern))
        blur_video_paths = sorted(glob.glob(blur_video_path_pattern))

        if not sharp_video_paths or not blur_video_paths:
            raise ValueError('no video was found')

        if len(sharp_video_paths) != len(blur_video_paths):
            raise ValueError('number of sharp videos must equal to number \
            of blur video')

        dataset = zip(sharp_video_paths, blur_video_paths)

        for idx, (sharp_video_path, blur_video_path) in enumerate(dataset):
            self.build_one_video(sharp_video_path, blur_video_path, dst, idx)
