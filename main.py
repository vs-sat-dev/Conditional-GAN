import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from train import gen_disc_train, classifier_train


IMG_SIZE = 64

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


if __name__ == '__main__':
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [50000, 10000])

    print('Generator and Discriminator train')
    generator = gen_disc_train(dataset_train)
    print('Classification with MNIST dataset')
    classifier_mnist_data = classifier_train(dataset_train, dataset_test)
    print('Classification with generated MNIST dataset')
    classifier_generator_data = classifier_train(dataset_train, dataset_test, generator)
