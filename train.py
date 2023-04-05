import torch
from utils import saveCheckpoint, loadCheckpoint, saveSomeExamples, showSomeExamples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def trainMethod(
    disc, gen, loader, opt_disc, optGenerator, l1_loss, bce, gScaler, dScaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            yFake = gen(x)
            dReal = disc(x, y)
            dRealLoss = bce(dReal, torch.ones_like(dReal))
            dFake = disc(x, yFake.detach())
            dFakeLoss = bce(dFake, torch.zeros_like(dFake))
            dLoss = (dRealLoss + dFakeLoss) / 2

        disc.zero_grad()
        dScaler.scale(dLoss).backward()
        dScaler.step(opt_disc)
        dScaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            dFake = disc(x, yFake)
            gFakeLoss = bce(dFake, torch.ones_like(dFake))
            L1 = l1_loss(yFake, y) * config.L1_LAMBDA
            gLoss = gFakeLoss + L1

        optGenerator.zero_grad()
        gScaler.scale(gLoss).backward()
        gScaler.step(optGenerator)
        gScaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(dReal).mean().item(),
                D_fake=torch.sigmoid(dFake).mean().item(),
            )


def main():
    disc = Discriminator(inChannels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    optimizeDiscrim = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    optimizeGenerator = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1Loss = nn.L1Loss()

    if config.LOAD_MODEL:
        loadCheckpoint(
            config.CHECKPOINT_GEN, gen, optimizeGenerator, config.LEARNING_RATE,
        )
        loadCheckpoint(
            config.CHECKPOINT_DISC, disc, optimizeDiscrim, config.LEARNING_RATE,
        )

    trainingDataset = MapDataset(root_dir=config.TRAIN_DIR)
    trainingLoader = DataLoader(
        trainingDataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    gScaler = torch.cuda.amp.GradScaler()
    dScaler = torch.cuda.amp.GradScaler()
    truthDataset = MapDataset(root_dir=config.VAL_DIR)
    truthLoader = DataLoader(truthDataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        trainMethod(
            disc, gen, trainingLoader, optimizeDiscrim, optimizeGenerator, L1Loss, BCE, gScaler, dScaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            saveCheckpoint(gen, optimizeGenerator, filename=config.CHECKPOINT_GEN)
            saveCheckpoint(disc, optimizeDiscrim, filename=config.CHECKPOINT_DISC)
            print("%d of %d" % (epoch+1, config.NUM_EPOCHS))

        saveSomeExamples(gen, truthLoader, epoch, folder=config.EVAL_DIR)


def previewModel():
    assert config.LOAD_MODEL

    disc = Discriminator(inChannels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    optimizeDiscriminator = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    optimizeGenerator = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    if config.LOAD_MODEL:
        loadCheckpoint(
            config.CHECKPOINT_GEN, gen, optimizeGenerator, config.LEARNING_RATE,
        )
        loadCheckpoint(
            config.CHECKPOINT_DISC, disc, optimizeDiscriminator, config.LEARNING_RATE,
        )

    truthDataset = MapDataset(root_dir=config.VAL_DIR)
    truthLoader = DataLoader(truthDataset, batch_size=1, shuffle=False)
    trainingDataset = MapDataset(root_dir=config.TRAIN_DIR)
    trainingLoader = DataLoader(
        trainingDataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    showSomeExamples(gen, truthLoader, 1, folder=config.PEEP_DIR, affix='aUnseen - ')
    showSomeExamples(gen, trainingLoader, 1, folder=config.PEEP_DIR, affix='bTruth - ')
    return


if __name__ == "__main__":
    if config.OBSERVE_MODEL:
        print("Not Updating Model")
        previewModel()
    else:
        print("Training Model")
        main()

