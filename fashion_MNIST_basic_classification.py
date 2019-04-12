#implementação do tutorial disponível em https://colab.research.google.com/drive/1cUQ5pYRCfjbd_-RbJ1NH-lRop6vhhP2x#scrollTo=S5Uhzt6vVIB2

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# Bibliotecas de ajuda
import fashion_MNIST_dataset as ds
import fashion_MNIST_model as model

class fashion_MNIST_basic_classification:
    def __init__(self):
        self.dataset = ds.fashion_MNIST_dataset()
        self.model = model.fashion_MNIST_trained_model()

#####################
        
bc = fashion_MNIST_basic_classification()


(train_images, train_lables), (test_images, test_lables) = bc.dataset.get_train_and_test_datasets()

train_images = bc.dataset.normalize_data(train_images)
test_images = bc.dataset.normalize_data(test_images)

bc.dataset.plot_image(1)

bc.dataset.plot_images(25)

bc.model.train(train_images, train_lables)

bc.model.save()

bc.model.evaluate(test_images, test_lables)

bc.model.predict(test_images, test_lables)