import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        imageFile = self.list_files[index]
        imagePath = os.path.join(self.root_dir, imageFile)
        image = np.array(Image.open(imagePath))
        inputImage  = image[:, config.IMAGE_HALF_X:, :]
        targetImage = image[:, :config.IMAGE_HALF_X, :]

        augmentations = config.bothTransform(image=inputImage, image0=targetImage)
        inputImage   = augmentations["image"]
        targetImage  = augmentations["image0"]

        inputImage  = config.transformOnlyInput(image=inputImage)["image"]
        targetImage = config.transformOnlyOutput(image=targetImage)["image"]

        return inputImage, targetImage


if __name__ == "__main__":
    dataset = MapDataset(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=15)
    for x, y in loader:
        print(x.shape)
        save_image(x, "Data/Test/x.png")
        save_image(y, "Data/Test/y.png")

        from PIL import Image
        imageX = Image.open("Data/Test/x.png")
        imageY = Image.open("Data/Test/y.png")
        combined = Image.new("RGB", (2*imageX.size[0], imageX.size[1]), (250, 250, 250))
        combined.paste(imageX, (0, 0))
        combined.paste(imageY, (imageX.size[0], 0))
        combined.save("Data/Test/x-y.png")

        import sys
        sys.exit()

