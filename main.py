#!/usr/bin/env python3

from functools import cache

from PIL import Image
import torch
from torch import nn

IMAGE_PATH = 'train-data/cake.png'
IMAGE_SIZE = 512
MODEL_PATH = 'out/model.pth'
MODEL_CKPT_PATH = 'out/model-{epoch:03d}.pth'
OUT_IMAGE_PATH = 'out/out.png'
OUT_IMAGE_CKPT_PATH = 'out/out-{epoch:03d}.png'

LEARNING_RATE = 1e-3
N_EPOCHS = 100
CKPT_PERIOD = 10


@cache
def get_pixel_coords():
    result = torch.empty((IMAGE_SIZE*IMAGE_SIZE, 2), dtype=torch.float32)
    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            i = y*IMAGE_SIZE + x
            result[i, 0] = (x+0.5) / IMAGE_SIZE
            result[i, 1] = (y+0.5) / IMAGE_SIZE

    return result


def load_image_data():
    with Image.open(IMAGE_PATH) as im:
        assert im.size == (IMAGE_SIZE, IMAGE_SIZE)
        assert im.mode == 'RGB'

        return torch.as_tensor(im.getdata(), dtype=torch.float32).div_(255)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def gen_image(model):
    with torch.no_grad():
        image_data = model(get_pixel_coords())
        image_data = image_data.clamp_(0, 1).mul_(255).round_().byte() \
            .view(IMAGE_SIZE, IMAGE_SIZE, 3)
        return Image.fromarray(image_data.numpy(), mode='RGB')


def main():
    x = get_pixel_coords()
    y = load_image_data()

    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*10, steps_per_epoch=1, epochs=N_EPOCHS,
        verbose=True)

    for epoch in range(N_EPOCHS):
        x_perturbed = torch.rand_like(x).sub_(0.5).div_(IMAGE_SIZE).add_(x)

        pred = model(x_perturbed)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch+1}/{N_EPOCHS}: loss = {loss.item():.6g}',
              flush=True)

        if (epoch+1) % CKPT_PERIOD == 0:
            torch.save(model.state_dict(),
                       MODEL_CKPT_PATH.format(epoch=epoch+1))
            gen_image(model).save(OUT_IMAGE_CKPT_PATH.format(epoch=epoch+1))

    torch.save(model.state_dict(), MODEL_PATH)
    gen_image(model).save(OUT_IMAGE_PATH)


if __name__ == '__main__':
    main()
