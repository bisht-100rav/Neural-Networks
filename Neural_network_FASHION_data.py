import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']  # type of clothes present inside the dataset

train_images = train_images/255.0   # make images pixel by pixel
test_images = test_images/255.0

#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()


#Starting Model

model = keras.Sequential([                            # Sequential provides Sequence to the layer
    keras.layers.Flatten(input_shape=(28,28)),        #input layer
    keras.layers.Dense(128, activation="relu"),        # hidden layer giving 128 neurons
    keras.layers.Dense(10, activation="softmax")      #estimation
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 5)    # epochs gives same images in different to model if models chooses wrong answer to image


#test_Loss, test_acc= model.evaluate(test_images, test_labels)      # for accuracy and loss
#print("Tested Acc", test_acc)

prediction = model.predict(test_images)
#print(class_names[np.argmax(prediction[0])])


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

#print(class_names[np.argmax(prediction[0])])
