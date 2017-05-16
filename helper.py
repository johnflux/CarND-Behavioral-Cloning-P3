import keras
import random
import matplotlib.pyplot as plt
import numpy as np

def showRandomImages(samples, f = None):
    for sample in samples:
        sample.showImage(f)
    exit(1)

def showDrivingAngles(samples, title="samples"):
    plt.hist([sample.driving_angle for sample in samples ], 16)
    plt.title("Driving angle distribution in " + title)
    plt.show()

def duplicateSamplesToRebalanceByDrivingAngle(samples):
    # Bin the data, returning an array of which bin each sample is in
    num_bins = 16
    indexes = np.digitize([sample.driving_angle for sample in samples], np.arange(-1, 1, 2/num_bins))
    # Find how many samples are in the largest bin
    largest_bin_count = np.max(np.bincount(indexes))
    print(largest_bin_count)
    rebalanced_samples = []
    for j in range(num_bins):
        bin_indexes = np.where(indexes==(j+1))[0]
        for i in range(largest_bin_count):
            rebalanced_samples.append(samples[bin_indexes[i % len(bin_indexes)]])
    random.shuffle(rebalanced_samples)
    return rebalanced_samples

def showLayerOutput(sample, inputs, outputs):
    model = keras.models.Model(inputs=inputs, outputs=output)
    sample.showImage()
    croppedimage = model.predict(np.array([sample.getImage()]))[0]
    print(croppedimage)
    plt.imshow(croppedimage)
    plt.show()

class DebugCallback(keras.callbacks.Callback):
    def __init__(self, train_samples, model):
        super().__init__()
        self.train_samples = train_samples
        self.model = model
    def on_epoch_end(self, batch, logs={}):
        print()
        print(logs)
        print("Should be: ", [sample.driving_angle for sample in self.train_samples])
        print("Predicted: ", [x[0] for x in self.model.predict(np.array([sample.getImage() for sample in self.train_samples]))]) #Print predicted driving angle for first example
        print()
