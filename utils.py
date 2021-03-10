from plantcv import plantcv as pcv
import cv2
from skimage.transform import match_histograms


class Cropper:

    # roi: [x, y, h, w]
    def __init__(self, roi):
        self.roi = roi

    def __call__(self, img):
        if not self.roi:
            return img
        return img[self.roi[0]: self.roi[0] + self.roi[2],
                  self.roi[1]: self.roi[1] + self.roi[3], :]


def flip(pattern):
    img_paths = sorted(glob.glob(pattern))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.flip(img, 1)
        cv2.imwrite(img_path, img)
    

def create_mask(img, mask, x_start, y_start, x_shift, y_shift, start_idx,
                del_list):
    dimensions = [20, 20] 

    chips = []

    row_total = 6
    col_total = 4

    for i in range(row_total):
        for j in range(col_total):
            chips.append(pcv.roi.rectangle(img=img, x=x_start + j * x_shift,
                                           y=y_start + i * y_shift,
                                           w=dimensions[0], h=dimensions[1]))

    for i in del_list:
        del chips[i]

    for chip in chips:
        mask = cv2.drawContours(mask, chip[0], -1, 1, -1)
        start_idx += 1

    mask = mask
    return mask 


def correct_color_plantcv(
        target_img,
        source_img,
        mask,
        output_directory='color_correction'):

    pcv.params.debug = "plot"

    target_matrix, source_matrix, transformation_matrix, corrected_img =\
            pcv.transform.correct_color(target_img=target_img,
                                        target_mask=mask,
                                        source_img=source_img,
                                        source_mask=mask,
                                        output_directory=output_directory)

    pcv.plot_image(mask)
    pcv.transform.quick_color_check(source_matrix=source_matrix,
            target_matrix=target_matrix, num_chips=24)

    return corrected_img


def correct_color_scikit(sharp, blur):
    sharp = cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR)
    blur = cv2.cvtColor(blur, cv2.COLOR_RGB2BGR)

    return match_histograms(blur, sharp, multichannel=True)
