import matplotlib.pyplot as plt
import numpy as np
import skimage.data

from keras import preprocessing
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_mean


from Hopfield import HopfieldNN


ALPHABET = ['С', 'Р', 'Г', 'Н', 'К']


def load_image(path):
    images = []
    for l in ALPHABET:
        p = path + l + '.png'
        img = preprocessing.image.load_img(p, target_size=(28, 28), color_mode="grayscale")
        img = preprocessing.image.img_to_array(img).reshape(28, 28)
        images.append(img)
    images = np.array(images)
    return images


def corrupt(data, corruption_level):
    corrupted = np.copy(data)
    inv = np.random.binomial(n=1, p=corruption_level*0.5, size=len(data))
    for i, v in enumerate(data):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def preprocess(img, w=128, h=128):
    # Resize image
    img = resize(img, (w, h), mode='reflect')
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2 * (binary * 1) - 1  # Boolean to int

    # Reshape
    flatten = np.reshape(shift, (w * h))
    return flatten


def reshape(data):
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))


def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


def plot_diff(diff, c, corruption=True):
    plt.plot(c, diff)
    if corruption:
        plt.xlabel("Corruption level")
    else:
        plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.show()


def get_diff(test_data, train_data):
    d = 0
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            d += abs(train_data[i][j] - test_data[i][j])
    return d / (len(train_data[0])*len(train_data))


if __name__ == "__main__":
    data = load_image(path='Data_bg/')

    data = [preprocess(d) for d in data]
    model = HopfieldNN(threshold=40)
    model.fit(train_data=data)

    #thresholds = [0, 20, 40, 60, 80, 100]
    cs = [0, .2, .4, .6, .8, .9, .95, .98, .99, 1]
    diff = []

    test = [corrupt(d, .2) for d in data]
    predicted = model.predict(test_data=test)
    plot(data, test, predicted)
