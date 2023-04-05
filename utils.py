import random

import torch
import config
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
import os


def showSomeExamples(gen, val_loader, epoch, folder, affix=""):
    loop = tqdm(val_loader, leave=True)
    folder = folder + '/' + affix if affix != "" else folder + '/'
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)

        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(x * 0.5 + 0.5, folder + "x.png")
            save_image(y_fake, folder + "y.png")
            combineTwoImages(folder, epoch)

        gen.train()
        epoch += 1

    os.remove(folder + 'x.png')
    os.remove(folder + 'y.png')
    return


def combineTwoImages(folder, epoch):
    imageX = Image.open(folder + 'x.png')
    imageY = Image.open(folder + 'y.png')
    combined = Image.new("RGB", (2 * imageX.size[0], imageX.size[1]), (250, 250, 250))
    combined.paste(imageX, (0, 0))
    combined.paste(imageY, (imageX.size[0], 0))
    combined.save(folder + "%d.png" % epoch)


def saveSomeExamples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def saveCheckpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("*** Saving checkpoint ***")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def loadCheckpoint(checkpoint_file, model, optimizer, lr):
    print("*** Loading checkpoint ***")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint,
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

