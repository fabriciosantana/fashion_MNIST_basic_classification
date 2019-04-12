from tensorflow import keras
import fashion_MNIST_plot as plot

class fashion_MNIST_dataset:
    def __init__(self):
        self.__class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print(">>> Downloadind MNIST dataset from keras:")
        print(">>> https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles")
        print(">>> Loadind dataset")
        (self.__train_images, self.__train_labels), (self.__test_images, self.__test_labels) = keras.datasets.fashion_mnist.load_data()
        print(">>> Dataset loaded into training sets (train_images, train_labels) and testing sets (test_images, test_labels) arrays")
        self.__plot__ = plot.fashion_MNIST_plot()

    def get_train_and_test_datasets(self):
        print(">>> Number of images in training set")
        print(len(self.__train_labels))
        print(self.__train_labels)
        #print(train_images)
        print(self.__train_images.shape)
        print(">>> Number of images in testing set")
        print(len(self.__test_labels))
        #print(test_images)
        print(self.__test_images.shape)
        return (self.__train_images, self.__train_labels), (self.__test_images, self.__test_labels)
     
    def get_class_names(self):
        print(">>> Class (labels) names")
        return self.__class_names

    def normalize_data(self,dataset):
        print(">>> As images são 28x28 Numpy arrays, onde os valores de cada pixel variam entre 0 e 255. Os labels são um vetor de inteiros, que varia entre 0 e 9. Estes correspondem à classe de roupa que a imagem representa")
        return dataset / 255.0
    
    def plot_image(self, image_index):
        self.__plot__.plot_image_from_dataset(self.__train_images, image_index)
        
    def plot_images(self, qtd_images):
        self.__plot__.plot_images_from_dataset(qtd_images, self.__train_images, self.__class_names, self.__train_labels)