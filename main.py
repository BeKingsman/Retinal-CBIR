from PIL import *
import cv2
import os
import glob
from skimage import io
import random
import numpy as np
from skimage.feature import texture as sktex
import time
import threading
import pickle
import math
from db import add_img_feature_to_db, get_db_features, Recreate_table
import dictances
import threading
'''
part-1 will be to generate feature vectors from input images
part-2 will be similarity matching
'''

img_size = 600

weight_labels = [
    'color histogram', 'energy', 'contrast', 'homogeneity', 'ASM',
    'dissimilarity', 'correlation', 'entropy'
]
# weights_array = [0.7,(0.7/7),(0.7/7),(0.7/7),(0.7/7),(0.7/7),(0.7/7),(0.7/7)]
weights_array = [
    0.21999623, 0.72999698, 0.66998567, 0.79998718, 0.38999397, 0.49001734,
    0.63998794, 0.57000377
]
NO_OF_THREADS = 10

temp_database_features = []


def custom_show(img):
    cv2.imshow("Converted", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pre_process_img(img):
    # Optic nerve on Left side
    template = cv2.imread('util/optic_nerve.jpeg')
    heat_map = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)
    if (x > img.shape[1] / 2):
        img = cv2.flip(img, 1)
    # resizing image

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # custom_show(img)
    return img


def local_histogram_customize(hist):
    n = hist.shape[0]
    temp = hist
    # print(hist,end="\n\n")
    x = hist[0]
    for i in range(1, n):
        temp[i] += x / 2
        x = (x / 2) + hist[i]

    i = n - 2
    x = hist[n - 1]
    while (i >= 0):
        temp[i] += x / 2
        x = (x / 2) + hist[i]
        i -= 1
    # print(temp,end="\n\n")
    return temp


def get_local_histogram(img):
    res = np.array([])

    for binx in range(3):
        for biny in range(3):
            for i in range(3):
                hist = cv2.calcHist([
                    img[int((img_size / 3) * binx):int((img_size / 3) *
                                                       (binx + 1)),
                        int((img_size / 3) * biny):int((img_size / 3) *
                                                       (biny + 1)), :]
                ], [i], None, [16], [0, 256])
                # hist = cv2.normalize(hist, dst=np.array([])).flatten()
                # hist = local_histogram_customize(hist.flatten()/100000)
                hist = hist.flatten() / 100000
                res = np.concatenate((res, hist))
    return res


#Global Histogram
# def get_local_histogram(img):
#     img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     res=np.array([])
#     for i in range(3):
#         hist = cv2.calcHist([img], [i], None, [64], [0,256])
#         # hist = cv2.normalize(hist, dst=np.array([])).flatten()
#         hist = hist.flatten()/10000
#         # hist=local_histogram_customize(hist)
#         res=np.concatenate((res, hist))
#     return res


def get_GLCM_features(image,
                      distances=(0),
                      angles=None,
                      levels=256,
                      symmetric=True,
                      normed=True,
                      features=None):
    if angles is None:
        angles = [0, np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4]

    if features is None:
        features = [
            "energy", "contrast", "homogeneity", "ASM", "dissimilarity",
            "correlation", "entropy"
        ]
    else:
        accepted_features = [
            "energy", "contrast", "homogeneity", "ASM", "dissimilarity",
            "correlation", "entropy"
        ]
        for f in features:
            if f not in accepted_features:
                raise Exception("Feature " + f +
                                "is not accepted in the set of features")

    image_glcm = sktex.greycomatrix(image,
                                    distances,
                                    angles,
                                    levels=levels,
                                    symmetric=symmetric,
                                    normed=normed)

    output_features = dict()
    for feature in features:
        if feature == "entropy":
            entropy = np.zeros((1, 4))
            for i in range(image_glcm.shape[0]):
                for j in range(image_glcm.shape[1]):
                    entropy -= image_glcm[i, j] * np.ma.log(image_glcm[i, j])
            output_features[feature] = entropy
        else:
            output_features[feature] = sktex.greycoprops(image_glcm, feature)

    return output_features


def get_glcm(img):
    img = img / 10
    output_features = get_GLCM_features(img.astype(np.uint8),
                                        distances=[10],
                                        levels=26)
    ans = np.array([])
    for key in output_features.keys():
        # print(key) ")
        ans = np.concatenate((ans, output_features[key][0]))
    # print("&&&&&&&&&&&&&&&&&&&\n")
    return ans


def get_fourier_descriptors(image):
    """ Function to find and return the	Fourier-Descriptor of the image contour
    :param image: OpenCV uint8 or float32 array_like
        Source image to compute the fourier descriptors on
    :return: array_like
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image_binary) = cv2.threshold(image, 127, 255,
                                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contour = []
    _, contour, hierarchy = cv2.findContours(image_binary, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE, contour)

    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result


def generate_features_from_img(img):
    # All features will be computed for this img
    img = pre_process_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_features = {}
    all_features["color_histogram"] = get_local_histogram(img)
    all_features["glcm"] = get_glcm(gray)
    # fourier_features_complex=get_fourier_descriptors(img)
    # all_features["fourier_descriptors"]=np.concatenate((fourier_features_complex.real,fourier_features_complex.imag))
    return all_features


def compute_database_features_util(img_li):
    for name in img_li:
        try:
            if (name != ".DS_Store"):
                image = cv2.imread('images/' + name)
                feature = generate_features_from_img(image)
                add_img_feature_to_db(
                    os.path.join(os.getcwd(), os.path.join("images", name)),
                    feature)
                print("Features Computed for " + name,
                      end="\n*************************\n\n")
        except Exception as e:
            print(str(e))


def compute_database_features():
    Recreate_table()
    threads = []
    img_li = os.listdir("images")
    n = len(img_li)
    seg = int(n / NO_OF_THREADS)
    for i in range(NO_OF_THREADS):
        t = threading.Thread(target=compute_database_features_util,
                             args=(img_li[i * seg:(i + 1) * seg], ))
        threads.append(t)
        t.start()
    for th in threads:
        th.join()


def distance_util(arr):
    return np.linalg.norm(arr)  #/math.sqrt(x)


def bhattacharyya_distance(a, b):
    ans = 0
    n = len(a)
    for bin in range(27):
        x = int(bin * (n / 27))
        y = int((bin + 1) * (n / 27))
        s1 = np.sum(a[x:y])
        s2 = np.sum(b[x:y])
        d1 = {}
        d2 = {}
        for i in range(x, y):
            d1[str(i)] = a[i] / s1
            d2[str(i)] = b[i] / s2
        temp = dictances.bhattacharyya(d1, d2)
        ans += temp
    return ans


def find_distance_array(query_features, db_features):
    distance_array = []
    color_histogram_distance = bhattacharyya_distance(
        query_features["color_histogram"], db_features["color_histogram"])
    distance_array.append(color_histogram_distance)
    glcm = (query_features["glcm"] - db_features["glcm"])
    for i in range(int(len(glcm) / 4)):
        dist = distance_util(glcm[4 * i:4 * (i + 1)])
        distance_array.append(dist)
    # fourier_descriptors_distance=np.linalg.norm(query_features["fourier_descriptors"]-db_features["fourier_descriptors"])
    distance_array = np.array(distance_array)
    return distance_array


def find_distance(query_feature, f, weights_array=weights_array):
    dist_2d = []
    res = []
    for ind in range(len(f)):
        x = f[ind]
        dist_arr = find_distance_array(query_feature, x)
        f[ind]["distance_array"] = dist_arr
        dist_2d.append(dist_arr)

    dist_2d = np.array(dist_2d)
    dist_coef = np.max(dist_2d, axis=0)

    for x in f:
        scaled_dist = x["distance_array"] / dist_coef
        dist = np.sum(scaled_dist * weights_array)
        res.append([dist, x["source"]])

    res.sort()
    return res


def query(query_img, n=50):
    print(query_img, end="\n\n\n")
    query_img = cv2.imread(query_img)
    print("Generating Query Image Features")
    query_feature = generate_features_from_img(query_img)
    print("Retriving DB Features")
    f = get_db_features()
    res = find_distance(query_feature, f)
    if (n and n < len(res)):
        res = res[:n]
    for im in res:
        print("Distance from " + im[1] + " is : " + str(im[0]))
    return res


if __name__ == "__main__":
    compute_database_features()
