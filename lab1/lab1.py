import numpy as np
import tensorflow

from keras import Sequential, losses, optimizers
from keras import layers
import matplotlib.pyplot as plt
from keras.src.saving.saving_api import save_model, load_model


def func(x):
    n = len(x)
    y = np.zeros((n,))
    for i in range(n):
        y[i] = 3 * x[i][0] ** 2 - 4 * x[i][0] * x[i][1]
    return y


def create_data(n):
    x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 80, 100]

    x_train = np.random.choice(x_values, (n, 2))
    x_test = []
    for i in range(0, 101):
        x_test.append([i, i])
    x_test = np.array(x_test)
    y_train = func(x_train)
    y_test = func(x_test)

    return x_train, y_train, x_test, y_test


def create_model(hidden_units, activation_func):
    model = Sequential()
    model.add(layers.Dense(hidden_units, activation=activation_func, input_shape=(2,)))
    model.add(layers.Dense(hidden_units // 4, activation=activation_func))
    model.add(layers.Dense(hidden_units // 8, activation=activation_func))
    model.add(layers.Dense(hidden_units // 16, activation=activation_func))
    model.add(layers.Dense(1, activation='linear'))

    return model


def train_model(model, EPOCHS, x_train, y_train, x_test, y_test):
    mse = losses.mean_squared_error
    model.compile(optimizers.Adam(learning_rate=0.01), loss=mse)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=256)
    plot_history(history, EPOCHS)
    return model


def test_model(model, x_test):
    y_pred = model.predict(x_test)

    y_exact = func(x_test)
    # y_pred = scale_y.inverse_transform(y_pred)
    x1 = []
    for i in range(len(x_test)):
        x1.append(x_test[i][0])
    plt.plot(x1, y_pred, label="Predicted values")
    plt.plot(x1, y_exact, label="Exact values")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    print("exact values:", y_exact)
    print("predicted values", y_pred)


def plot_history(model, epochs):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_summary(loss):
    f = open('stats/' + str(NS) + '.txt', 'w')
    f.write(str(loss))
    f.close()


def plot_stats():
    print("loss for:")
    # Plot test loss
    stats = []

    f = open('stats/relu.txt', 'r')
    loss = float(f.read())
    stats.append(loss)
    print("relu: " + str(loss))
    f.close()

    f = open('stats/sigmoid.txt', 'r')
    loss = float(f.read())
    stats.append(loss)
    print("sigmoid: " + str(loss))
    f.close()

    f = open('stats/softmax.txt', 'r')
    loss = float(f.read())
    stats.append(loss)
    print("softmax: " + str(loss))
    f.close()

    f = open('stats/tanh.txt', 'r')
    loss = float(f.read())
    stats.append(loss)
    print("tanh: " + str(loss))
    f.close()

    f = open('stats/linear.txt', 'r')
    loss = float(f.read())
    stats.append(loss)
    print("linear: " + str(loss))
    f.close()

    plt.title("Loss")
    plt.plot(['relu', 'sigmoid', "softmax", 'tanh', 'linear'], stats)
    plt.xlabel("Activation function")
    plt.ylabel("Test loss")
    plt.show()


if __name__ == '__main__':
    # Инициализация констант
    NS = 2048
    EPOCHS = 1000
    DATA_SIZE = 100000
    ACTIVATION_FUNC = 'relu'

    x_train, y_train, x_test, y_test = create_data(DATA_SIZE)
    my_model = create_model(NS, ACTIVATION_FUNC)
    trained_model = train_model(my_model, EPOCHS, x_train[:int(DATA_SIZE/8)], y_train[:int(DATA_SIZE/8)],
                                x_train[int(DATA_SIZE/8):], y_train[int(DATA_SIZE/8):])
    # my_model = load_model('models/model-16.h5')
    test_model(my_model, x_test)
    test_loss = my_model.evaluate(x_test, y_test)
    name = 'models/model'
    name = name + '-' + str(NS) + '.h5'
    save_summary(test_loss)
    save_model(my_model, name)
    #plot_stats()
