import numpy as np
import matplotlib.pyplot as plt

def show_images(x_train, y_train):
    """
    Function to plot the MNIST dataset.
    """
    plt.figure(figsize=(20,5))
    # convert labels into integers
    y_train = [int(numeric_string) for numeric_string in y_train]
    for index, (img, label) in enumerate(zip(x_train[0:4], y_train[0:4])):
        plt.subplot(1, 4, index + 1)
        plt.imshow(np.reshape(img, (28,28)), cmap=plt.cm.binary_r)
        plt.title('Label: %i\n' % label, fontsize = 18)
    


