import numpy as np
import skimage.transform
import skimage.io


def preprocess_img(img, size, save_test_img=False):
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
