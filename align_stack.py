import cv2
import numpy as np
import glob
import os
from rawkit.raw import Raw
from rawkit.options import WhiteBalance

cwd = os.getcwd()


# rawkit not working for some reason...
def read_cr2_image(file_path):
    with Raw(filename=file_path) as raw:
        raw.options.white_balance = WhiteBalance.camera
        buffered_image = np.array(raw.to_buffer())

    rgb_image = buffered_image.reshape(raw.metadata.height, raw.metadata.width, -1)
    return rgb_image


def align_images(im1, im2, max_warp=10):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    _, warp_matrix = cv2.findTransformECC(
        im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, max_warp
    )
    aligned_image = cv2.warpAffine(
        im2,
        warp_matrix,
        (im1.shape[1], im1.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    return aligned_image


def stack_images(image_files):
    base_image = cv2.imread(image_files[0])
    image_stack = np.array(base_image, dtype=np.float32)

    for image_file in image_files[1:]:
        image = cv2.imread(image_file)

        aligned_image = align_images(base_image, image)

        image_stack += aligned_image

    stacked_image = cv2.normalize(
        image_stack, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return stacked_image


if __name__ == "__main__":
    image_files = glob.glob(os.path.join(cwd, "*.CR2"))
    stacked_image = stack_images(image_files)
    cv2.imwrite(os.path.join(cwd, "stacked", "stacked_image.jpg"), stacked_image)
