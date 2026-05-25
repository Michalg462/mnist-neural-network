from torch.utils.data import random_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# This class allows for Binarization of an image.
# Can be used as a transformation in DataPrep
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor > self.threshold).float()

# This class implements a simple color negative operation.
class Negative(object):
    def __call__(self, tensor):
        return 1 - tensor

# Main class used to prepare data for the models.
class DataPrep:
    # Prepares images for the model training.
    def prepare_training_data(path):
        # Image transformation definition
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Loading the dataset from the given path
        dataset = datasets.ImageFolder(
            root=path,
            transform=transform
        )
        # data split - 90/10 for training/testing
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # preparing the loader objects used in the batch learning process
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        return train_loader, val_loader

    # This function returns the cropped out center of the image, by default 400x400 pixels, but can be changed
    @staticmethod
    def crop_center(img, crop_size=400):
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return img.crop((left, top, right, bottom))

    # This function returns a single image that can be passed to a model for evaluation
    # preview - should the image preview be rendered after transformations
    # flatten - should the image be returned as a 784 X 1 or 28x28x1 format
    def prepare_test_image(self, path, preview=False, flatten=True):

        # Check if the image under the given path exists
        if not isinstance(path, str):
            raise ValueError(f"Invalid path '{path}': Expected a file path as a string.")

        # Preparation of the transformations that shall be applied
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.GaussianBlur(3, 0.5),
            transforms.RandomAutocontrast(p=1.0),
            transforms.ToTensor(),
            # Binarize(0.3),
            Negative(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Main image processing pipeline
        img = Image.open(path)
        img = self.crop_center(img, 2200)
        # Applies the metadata transformation info
        img = ImageOps.exif_transpose(img)
        img = transform(img)

        if preview:
            self.show_tensor_image(img)

        if flatten:
            img = img.view(1, -1)

        return img

    # Displays the image preview in the form of the plot. Can be used to measure specific pixel values
    @staticmethod
    def show_tensor_image(tensor):
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        tensor = tensor * 0.5 + 0.5
        img = tensor.numpy()

        plt.imshow(img, cmap="gray")
        plt.title("Transformed picture")
        plt.axis("off")
        plt.show()