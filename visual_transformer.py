import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
from torchvision import transforms, models
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")


class DICOMDataset(Dataset):
    def __init__(self, data, root_dir, target_column=None, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        study_id = str(row["study_id"])
        image_path = self.find_dicom_file(study_id)

        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Missing file for study ID {study_id}. Skipping.")
            return None, None

        try:
            dicom = pydicom.dcmread(image_path)
            image = dicom.pixel_array
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")

            if self.transform:
                image = self.transform(image)

            if self.target_column:
                label = row[self.target_column]
                label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
                label = label_map.get(label, -1)
                return image, torch.tensor(label, dtype=torch.long)
            else:
                return image
        except Exception as e:
            print(f"Error reading file for study ID {study_id}: {e}")
            return None, None

    def find_dicom_file(self, study_id):
        for root, _, files in os.walk(self.root_dir):
            if study_id in root:
                for file in files:
                    if file.endswith(".dcm"):
                        return os.path.join(root, file)
        return None


class FilteredDataloader:
    @staticmethod
    def filter_invalid_samples(dataset):
        valid_samples = [dataset[i] for i in range(len(dataset)) if dataset[i][0] is not None]
        print(f"Filtered out invalid samples: {len(dataset) - len(valid_samples)} removed.")
        return valid_samples

    @staticmethod
    def create_dataloader(dataset, batch_size, shuffle=True):
        valid_samples = FilteredDataloader.filter_invalid_samples(dataset)
        return DataLoader(valid_samples, batch_size=batch_size, shuffle=shuffle)


class VisionTransformerTrainer:
    def __init__(self, root_dir, train_labels, target_column, model_path, batch_size=16, num_classes=3, epochs=25, output_dir="plots"):
        self.root_dir = root_dir
        self.train_labels = train_labels
        self.target_column = target_column
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"loss": [], "accuracy": []}
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\nInitializing training for target column: {self.target_column}...")

        self.train_loader, self.test_loader = self.load_and_prepare_data()
        self.model = self.setup_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        self.train_model()
        self.plot_metrics()

    def setup_transformations(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def load_and_prepare_data(self):
        print("\n=== Loading and Preparing Data ===")
        full_data = pd.read_csv(self.train_labels)
        valid_data = full_data[full_data[self.target_column].notnull()]
        print(f"Valid Dataset Size: {len(valid_data)}")

        if len(valid_data) == 0:
            raise ValueError(f"No valid samples found for target column: {self.target_column}.")

        train_size = int(len(valid_data) * 0.7)
        train_data = valid_data[:train_size]
        test_data = valid_data[train_size:]

        transform = self.setup_transformations()
        train_dataset = DICOMDataset(data=train_data, root_dir=self.root_dir, target_column=self.target_column, transform=transform)
        test_dataset = DICOMDataset(data=test_data, root_dir=self.root_dir, target_column=self.target_column, transform=transform)

        train_loader = FilteredDataloader.create_dataloader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = FilteredDataloader.create_dataloader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def setup_model(self):
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        return model.to(self.device)

    def train_model(self):
        print("\nStarting Training...")
        patience = 5
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in self.train_loader:
                if images is None or labels is None:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            self.history["loss"].append(epoch_loss)
            self.history["accuracy"].append(epoch_accuracy)

            print(f"Epoch [{epoch + 1}/{self.epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Improved model saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
                print(f"No improvement in loss for {patience_counter} epoch(s).")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def plot_metrics(self):
        epochs = range(1, len(self.history["loss"]) + 1)
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["loss"], label="Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["accuracy"], label="Accuracy")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.output_dir, f"{self.target_column}_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")


class MultiColumnTrainer:
    def __init__(self, root_dir, train_labels, target_columns, model_dir, batch_size=16, num_classes=3, epochs=25):
        self.root_dir = root_dir
        self.train_labels = train_labels
        self.target_columns = target_columns
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs

        os.makedirs(self.model_dir, exist_ok=True)

        for target_column in self.target_columns:
            print(f"\nTraining model for target column: {target_column}")
            model_path = os.path.join(self.model_dir, f"{target_column}_vit_model.pth")
            try:
                VisionTransformerTrainer(
                    root_dir=self.root_dir,
                    train_labels=self.train_labels,
                    target_column=target_column,
                    model_path=model_path,
                    batch_size=self.batch_size,
                    num_classes=self.num_classes,
                    epochs=self.epochs
                )
            except ValueError as e:
                print(f"Skipping training for {target_column}: {e}")
