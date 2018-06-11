import numpy as np
import skimage.transform
import skimage.io
import skimage.color


def preprocess_img_red(img, size, save_test_img=False):
    x, y = size
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(
        img[150:400, :], size)  # Ucinamy sufit i UI gry

    # wycinamy tylko kanał RED, bo na nim najlepiej widać pociski
    img = np.resize(img[:, :, 0], (x, y, 1))
    img = np.squeeze(img, axis=2)

    if save_test_img:
        skimage.io.imsave('./test.png', img)

    return img


def preprocess_img_rgb(img, size, save_test_img=False):
    x, y = size
    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(
        img[150:400, :], size)  # Ucinamy sufit i UI gry
    img = skimage.color.rgb2grey(img)

    if save_test_img:
        skimage.io.imsave('./test.png', img)

    return img
