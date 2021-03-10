from plantcv import plantcv as pcv
import cv2
import numpy as np
from skimage.io import imread, imsave
from skimage import exposure
import glob
from skimage.transform import match_histograms


class MyColorCorrection:
    def __init__(self, source_img, target_img, mask=None, gamma=2.2):
        self.gamma = gamma

        self.target_img = self.rgb2lin(target_img)

        self.source_img = np.ones(
                (source_img.shape[0], source_img.shape[1], 4)
        )
        self.source_img[:, :, :3] = self.rgb2lin(source_img)

        if mask is None:
            mask = np.ones((self.target_img.shape[0],
                self.target_img.shape[1]))

        selection = np.where(mask > 0)
        self.Pt = self.flatten_image(self.target_img, mask)
        self.Ps = self.flatten_image(self.source_img, mask)
        self.T = self.getTransformationMatrix(0.1)

    def getTransformationMatrix(self, lmbd):
        Ps, Pt = self.Ps, self.Pt
        I = np.eye(4)

        return np.linalg.inv(Ps.transpose().dot(Ps) + lmbd *\
                I).dot(Ps.transpose().dot(Pt))

    def __call__(self, img):
        img_4c = np.ones((img.shape[0], img.shape[1], img.shape[2] + 1))
        img_4c[:, :, :3] = img

        img_lin = self.rgb2lin(img_4c)
        img_lin_flat = self.flatten_image(img_4c, np.ones(img.shape[:2]))
        img_lin_flat_corrected = img_lin_flat.dot(self.T)
        img_corrected = self.lin2rgb(
            self.restore_image(img_lin_flat_corrected, img.shape)
        )

        return img_corrected

    def rgb2lin(self, img):
        return 255 * ((img / 255.) ** (self.gamma))

    def lin2rgb(self, img):
        return 255 * ((img / 255.) ** (1 / self.gamma))

    @staticmethod
    def flatten_image(img, mask):
        xs, ys = np.where(mask > 0)
        _, _, c = img.shape

        d = len(xs)

        P = np.zeros((d, c))
        for i in range(c):
            P[:, i] = img[xs, ys, i].reshape((-1))

        return P

    @staticmethod
    def restore_image(flatten, shape):
        img = np.zeros(shape)
        for i in range(shape[2]):
            img[:, :, i] = flatten[:, i].reshape(shape[:2])

        return img




if __name__ == '__main__':
    sharp_img, _, _ = pcv.readimage(
            filename="res/colorboard/sharp.png"
    )
    blur_img, _, _ = pcv.readimage(
            filename="res/colorboard/blur.png"
    )

    mask = np.zeros(shape=np.shape(sharp_img)[:2], dtype = np.uint8())
    mask = create_mask(sharp_img, mask, 85, 115, 40, 42, 1, [23])
    mask = create_mask(sharp_img, mask, 300, 113, 41, 42, 1, [22, 21, 20, 18,
        17, 14])
    mask = mask * 10

    solver = MyColorCorrection(blur_img, sharp_img, np.ones((480, 480)), 1)

    img_corrected = solver.transform(blur_img)
    cv2.imwrite('corrected_all.png', img_corrected)
