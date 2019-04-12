# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class fashion_MNIST_plot:
    def __init__(self):
        print("Inicializing fashion_MNIST_plot")

    def plot_image_from_dataset(self, dataset, index):
        print(">>> Plotting image from training set")
        plt.figure()
        plt.imshow(dataset[index])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        
    def plot_images_from_dataset(self, qtd_images, dataset_values,labels_names, lables_values):
        plt.figure(figsize=(10,10))
        for i in range(qtd_images):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(dataset_values[i], cmap=plt.cm.binary)
            plt.xlabel(labels_names[lables_values[i]])
        plt.show()        