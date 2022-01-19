import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import Discriminator, Generator


transform = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NOISE_DIM = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10

if __name__ == '__main__':

    dataset_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model_gen = Generator(NOISE_DIM).to(DEVICE)
    model_disc = Discriminator().to(DEVICE)

    optim_gen = optim.Adam(params=model_gen.parameters(), lr=LEARNING_RATE)
    optim_disc = optim.Adam(params=model_disc.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    writer_real = SummaryWriter('../logs/real')
    writer_fake = SummaryWriter('../logs/fake')

    step = 0

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch + 1}')
        for data, labels in loader_train:
            labels = labels.to(DEVICE)
            real_images = data.to(DEVICE)
            fake_images = model_gen(torch.randn(real_images.shape[0], NOISE_DIM, 1, 1).to(DEVICE), labels)

            disc_real = model_disc(real_images, labels).reshape(-1)
            disc_fake = model_disc(fake_images, labels).reshape(-1)

            #print(f'real: {disc_real}')
            #print(f'fake: {disc_fake}')

            disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            optim_disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            optim_disc.step()

            gen_fake = model_disc(fake_images, labels).reshape(-1)
            gen_loss = criterion(gen_fake, torch.ones_like(gen_fake))
            optim_gen.zero_grad()
            gen_loss.backward()
            optim_gen.step()

            print(f'gen_loss: {gen_loss} disc_loss: {disc_loss} epoch: {epoch}')

            real_grid = torchvision.utils.make_grid(real_images)
            fake_grid = torchvision.utils.make_grid(fake_images)

            writer_real.add_image('Real', real_grid, global_step=step)
            writer_fake.add_image('Fake', fake_grid, global_step=step)

            step += 1

