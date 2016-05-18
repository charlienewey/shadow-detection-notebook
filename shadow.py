import numpy as np
import scipy.ndimage as ndi

from skimage.feature import local_binary_pattern

from skimage.filters import gaussian
from skimage.filters import threshold_otsu

from sklearn.cluster import MiniBatchKMeans as k_means

import warnings
warnings.simplefilter("ignore")


# Let's define a function to show images...
import math
def _imshow(width, axes, *images):
    from matplotlib import pyplot as plt
    fig = plt.figure()

    height = math.ceil(len(images) / float(width))
    for i in range(0, len(images)):
        im = images[i]
        ax = fig.add_subplot(height, width, i + 1)
        cax = ax.imshow(im, cmap=plt.cm.cubehelix)

        if not axes:
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def k_means_texture_cluster(image):
    n_clusters = 8

    # blur and take local maxima
    blur_image = gaussian(image, sigma=8)
    blur_image = ndi.maximum_filter(blur_image, size=3)

    # get texture features
    feats = local_binary_pattern(blur_image, P=40, R=5, method="uniform")
    feats_r = feats.reshape(-1, 1)

    # cluster the texture features, reusing initialised centres if already calculated
    km = k_means(n_clusters=n_clusters, batch_size=500)
    clus = km.fit(feats_r)

    # copy relevant attributes
    labels = clus.labels_

    # reshape label arrays
    labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])
    return labels


def k_means_classifier(image):
        n_clusters = 8

        # blur and take local maxima
        blur_image = gaussian(image, sigma=8)
        blur_image = ndi.maximum_filter(blur_image, size=3)

        # get texture features
        feats = local_binary_pattern(blur_image, P=40, R=5, method="uniform")
        feats_r = feats.reshape(-1, 1)

        # cluster the texture features
        km = k_means(n_clusters=n_clusters, batch_size=500)
        clus = km.fit(feats_r)

        # copy relevant attributes
        labels = clus.labels_
        clusters = clus.cluster_centers_

        # reshape label arrays
        labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

        # segment shadow
        img = blur_image.ravel()
        shadow_seg = img.copy()
        for i in range(0, n_clusters):
            # set up array of pixel indices matching cluster
            mask = np.nonzero((labels.ravel() == i) == True)[0]
            if len(mask) > 0:
                thresh = threshold_otsu(img[mask])
                shadow_seg[mask] = shadow_seg[mask] < thresh
        shadow_seg = shadow_seg.reshape(*image.shape)

        return shadow_seg


def k_means_variance_classifier(image):
        n_clusters = 8

        # blur and take local maxima
        blur_image = gaussian(image, sigma=8)
        blur_image = ndi.maximum_filter(blur_image, size=3)

        # get texture features
        feats = local_binary_pattern(blur_image, P=40, R=5, method="uniform")
        feats_r = feats.reshape(-1, 1)

        # cluster the texture features
        km = k_means(n_clusters=n_clusters, batch_size=500)
        clus = km.fit(feats_r)

        # copy relevant attributes
        labels = clus.labels_
        clusters = clus.cluster_centers_

        # reshape label arrays
        labels = labels.reshape(blur_image.shape[0], blur_image.shape[1])

        # segment shadow
        img = blur_image.ravel()
        shadow_seg = img.copy()
        for i in range(0, n_clusters):
            # set up array of pixel indices matching cluster
            mask = np.nonzero((labels.ravel() == i) == True)[0]
            if len(mask) > 0:
                if img[mask].var() > 0.005:
                    thresh = threshold_otsu(img[mask])
                    shadow_seg[mask] = shadow_seg[mask] < thresh
                else:
                    shadow_seg[mask] = 0
        shadow_seg = shadow_seg.reshape(*image.shape)

        return shadow_seg

