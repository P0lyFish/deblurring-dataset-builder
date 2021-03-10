import cv2
from scipy import signal
import numpy as np


class Homography:
    is_debugging = False

    def debug(self, src, src_corners, dst, dst_corners):
        if not self.is_debugging:
            return

        src_draw_corners = cv2.drawChessboardCorners(
                src,
                self.chessboard_size,
                src_corners,
                True
        )
        dst_draw_corners = cv2.drawChessboardCorners(
                dst,
                self.chessboard_size,
                dst_corners,
                True
        )

        cv2.imwrite('src_corners.png', src_draw_corners)
        cv2.imwrite('dst_corners.png', dst_draw_corners)

    def __init__(self, src, dst, chessboard_size):
        self.chessboard_size = chessboard_size

        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        src_found_corners, src_corners =\
                cv2.findChessboardCorners(src_gray, chessboard_size)
        dst_found_corners, dst_corners =\
                cv2.findChessboardCorners(dst_gray, chessboard_size)

        if (not src_found_corners) or (not dst_found_corners):
            raise ValueError('corners not found')

        self.debug(src, src_corners, dst, dst_corners)

        self.H, _ = cv2.findHomography(src_corners, dst_corners)

    def __call__(self, img):
        return cv2.warpPerspective(img, self.H, (img.shape[1], img.shape[0]))


if __name__ == '__main__':
    Homography.is_debugging = True
    src = cv2.imread('imgs/new/clock/sharp/00000050.png')
    dst = cv2.imread('imgs/new/clock/blur/00000050.png')
    h = Homography(src, dst, (7, 7))
    cv2.imwrite('warped.png', h(src))
