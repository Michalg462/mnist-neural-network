import torch.nn as nn
import torch
import torch.nn.functional as functional
from DataHandler import DataPrep

# Main network class. Defines the model parameters
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # First there are 2 convolutional layers. One 1 -> 32 layers, second 32 -> 64 layers.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        # Then we go to a 2 classic linear layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = functional.relu(self.conv1(x))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class responsible for model training, can get custom learning rate and number of epochs
class TrainCNN:
    def __init__(self, learning_rate=0.001, epochs=5):

        # Software info and device selection
        print(torch.__version__)
        print(torch.version.cuda)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using the device " + self.device + " for learning")

        # Creating a model object to be trained. Using standard configuration.
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

        self.train_loader, self.val_loader = DataPrep.prepare_training_data("archive/trainingSet/trainingSet")

    def start_training(self, output_path):
        # Starts the learning process, displays statistics after each epoch
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(self.train_loader)
            val_loss, val_acc = self.validate(self.val_loader)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        torch.save(self.model.state_dict(), output_path)
        print("Model saved.")

    # One epoch. Goes through all batches in the data set. Sums the statistics.
    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(loader.dataset)
        return total_loss / len(loader), accuracy

    # Validation of the model. Goes through the data and calculates statistics.
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(loader.dataset)
        return total_loss / len(loader), accuracy

# Class used for testing the model on a single given photo. Needs a path to image as an argument.
class TestCNN:
    def __init__(self, path):
        # Similar start to the training class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        # But this time enabling evaluation mode
        self.model.eval()

    # Helper method, prints the output statistics
    @staticmethod
    def print_logits_with_labels(logits):
        logits = logits.squeeze()
        probs = functional.softmax(logits, dim=0)

        print("Class | Logit        | Probabilities")
        print("-------------------------------------------")
        for i in range(10):
            print(f"{i:5d} | {logits[i]:12.6f} | {probs[i]:.6f}")

    # Loads the images and passes it to the model. Prints the output.
    def test_example(self, path_to_image, show_example=False):
        img = DataPrep().prepare_test_image(path=path_to_image, preview=show_example, flatten=False)

        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = img.to(self.device)

        with torch.no_grad():
            out = self.model(img)
            pred = out.argmax(1).item()

        self.print_logits_with_labels(out)

        print("Prediction:", pred)
