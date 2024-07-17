import numpy as np
import random
import os
from PIL import Image


def hopfield(train_files, test_files, theta=0.5, time=10, size=(28, 28), threshold=0, current_path=None):
    print("Importing images and creating weight matrix...")
    num_files = 0
    for path in train_files:
        print(path)
        x = readImg2array(file=path, size=size, threshold=threshold)
        x_vec = mat2vec(x)
        if num_files == 0:
            w = create_W(x_vec)
            num_files = 1
        else:
            tmp_w = create_W(x_vec)
            w = w + tmp_w
            num_files += 1

    # Import test data
    counter = 0
    for path in test_files:
        y = readImg2array(file=path, size=size, threshold=threshold)
        oshape = y.shape
        print(y.shape)
        y_img = array2img(y)
        y_img.show()
        print("Imported test data")
        y_vec = mat2vec(y)
        print("Updating...")
        y_vec_after = update(w=w, y_vec=y_vec, theta=theta, time=time)
        y_vec_after = y_vec_after.reshape(oshape)
        print(y_vec_after.shape)
        if current_path is not None:
            outfile = current_path + "/after_" + str(counter) + ".png"
            array2img(y_vec_after, outFile=outfile)
        counter += 1


# Convert matrix to a vector
def mat2vec(x):
    m = x.shape[0] * x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i, j]
            c += 1
    return tmp1


# Create Weight matrix for a single image
def create_W(x):
    if len(x.shape) != 1:
        print("The input is not vector")
        return
    else:
        w = np.zeros([len(x), len(x)])
        for i in range(len(x)):
            for j in range(i, len(x)):
                if i == j:
                    w[i, j] = 0
                else:
                    w[i, j] = x[i] * x[j]
                    w[j, i] = w[i, j]
    return w


# Read Image file and convert it to Numpy array
def readImg2array(file, size, threshold=0):
    pilIN = Image.open(file).convert(mode="L")
    pilIN = pilIN.resize(size)
    imgArray = np.asarray(pilIN, dtype=np.int8)
    x = np.zeros(imgArray.shape, dtype=np.int8)
    x[imgArray > threshold] = -1
    x[x == 0] = 1
    return x


# Convert Numpy array to Image file
def array2img(data, outFile=None):
    # data is 1 or -1 matrix
    y = np.zeros(data.shape, dtype=np.uint8)
    # set main color here
    y[data == 1] = 255
    # set secondary color here
    y[data == -1] = 0
    img = Image.fromarray(y, mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


def corrupt(data, corruption_level):
    corrupted = np.copy(data)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(data))
    for i, v in enumerate(data):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


# Update
def update(w, y_vec, theta, time):
    y = corrupt(y_vec, .1)
    og = y
    for s in range(time):
        m = len(y)
        i = random.randint(0, m - 1)
        u = np.dot(w[i][:], y) - theta

        if u > 0:
            y[i] = 1
        elif u < 0:
            y[i] = -1
    print(og == y)
    return y


if __name__ == '__main__':
    ALPHABET = ['a']
    current_path = os.getcwd()
    train_paths = []
    test_paths = []

    for l in ALPHABET:
        train_paths.append(current_path + '/data/' + l + '.png')
        test_paths.append(current_path + '/data/' + l + '.png')

    hopfield(train_files=train_paths, test_files=test_paths, theta=0, time=100, size=(29, 29), threshold=0,
             current_path=current_path)
