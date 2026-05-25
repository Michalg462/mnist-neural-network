import torch.nn as nn
import torch
import torch.nn.functional as functional
from DataHandler import DataPrep

# Main network class. Defines the model parameters
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # The model contains one input layer, two hidden layer with consequently 256 and 128 neurons, and output layer.
        self.model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Class responsible for model training.
class TrainMLP:

    # The class can have custom learning rate and number of epochs
    def __init__(self, learning_rate=0.001, epochs=10):

        # First software info and device selection
        print(torch.__version__)
        print(torch.version.cuda)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using the device " + self.device + " for learning")

        # Here the model is prepared. Using the basic criterion and optimizer
        self.model = MLP().to(self.device)
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
            images = images.view(images.size(0), -1).to(self.device)
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
                images = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(loader.dataset)
        return total_loss / len(loader), accuracy

# Class used for testing the model on a single given photo. Needs a path to image as an argument.
class TestMLP:
    def __init__(self, path):
        # Similar start to the training class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP().to(self.device)
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
        img = DataPrep.prepare_test_image(DataPrep() ,path=path_to_image, preview=show_example)
        img = img.to(self.device)

        with torch.no_grad():
            out = self.model(img)
            pred = out.argmax(1).item()

        self.print_logits_with_labels(out)

        print("Prediction:", pred)