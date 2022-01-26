import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from gen_disc_models import Discriminator, Generator, initialize_weights
from classifier_model import Classifier


BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NOISE_DIM = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
IMG_SIZE = 64
NUM_CLASSES = 10
DISC_ITERATIONS = 5


def gradient_penalty(disc, real, fake, labels, device="cpu"):
    BS, C, H, W = real.shape
    alpha = torch.rand((BS, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = disc(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def gen_disc_train(dataset_train):

    loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model_gen = Generator(NOISE_DIM, NUM_CLASSES).to(DEVICE)
    model_disc = Discriminator(NUM_CLASSES, IMG_SIZE).to(DEVICE)

    initialize_weights(model_gen)
    initialize_weights(model_disc)

    optim_gen = optim.Adam(params=model_gen.parameters(), lr=LEARNING_RATE * 10.0, betas=(0.0, 0.9))
    optim_disc = optim.Adam(params=model_disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch + 1}')
        for batch_id, (data, labels) in enumerate(loader_train):
            labels = labels.to(DEVICE)
            real_images = data.to(DEVICE)
            fake_images = model_gen(torch.randn(real_images.shape[0], NOISE_DIM, 1, 1).to(DEVICE), labels)

            for _ in range(DISC_ITERATIONS):
                disc_real = model_disc(real_images, labels).reshape(-1)
                disc_fake = model_disc(fake_images, labels).reshape(-1)

                gp = gradient_penalty(model_disc, real_images, fake_images, labels, device=DEVICE)
                disc_loss = (-(torch.mean(disc_real) - torch.mean(disc_fake))) + 0.1 * gp

                optim_disc.zero_grad()
                disc_loss.backward(retain_graph=True)
                optim_disc.step()

            gen_fake = model_disc(fake_images, labels).reshape(-1)
            gen_loss = -torch.mean(gen_fake)
            optim_gen.zero_grad()
            gen_loss.backward()
            optim_gen.step()

    return model_gen


def classifier_train(dataset_train, dataset_test, generator_model=None):
    loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)

    model = Classifier().to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        # Train model
        for data, labels in loader_train:
            if generator_model is None:
                X = data.to(DEVICE)
            else:
                X = generator_model(torch.randn(data.shape[0], NOISE_DIM, 1, 1).to(DEVICE), labels.to(DEVICE))
            y = labels.to(DEVICE)

            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Check performance on the testing dataset
        with torch.no_grad():
            model.eval()
            preds, y = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
            for data, labels in loader_test:
                X = data.to(DEVICE)
                y = torch.cat((y, labels), dim=0)
                preds = torch.cat((preds, torch.argmax(model(X).cpu(), dim=1)), dim=0)
            accuracy = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average='weighted')
            print(f'Test accuracy_score: {accuracy}, f1_score: {f1}, epoch: {epoch}')
            model.train()

