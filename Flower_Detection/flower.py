import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from PIL import Image

import scipy.io
import os
from tqdm.auto import tqdm
import requests
import tarfile


# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================

def download_dataset():

    data_dir = "Flower_Detection/flower_data"

    image_folder_path = os.path.join(data_dir, "jpg")
    labels_file_path = os.path.join(data_dir, "imagelabels.mat")
    tgz_path = os.path.join(data_dir, "102flowers.tgz")

    if os.path.exists(image_folder_path) and os.path.exists(labels_file_path):
        print(f"Dataset already exists. Loading locally from '{data_dir}'.")
        return

    print("Dataset not found locally. Downloading...")

    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

    os.makedirs(data_dir, exist_ok=True)

    # --------------------------------------------------------
    # Download images
    # --------------------------------------------------------

    print("Downloading images...")

    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(tgz_path, "wb") as file:

        for data in tqdm(
            response.iter_content(chunk_size=1024),
            total=total_size // 1024,
            unit="KB"
        ):
            file.write(data)

    # --------------------------------------------------------
    # Extract images
    # --------------------------------------------------------

    print("Extracting files...")

    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir)

    # --------------------------------------------------------
    # Download labels
    # --------------------------------------------------------

    print("Downloading labels...")

    response = requests.get(labels_url)
    response.raise_for_status()

    with open(labels_file_path, "wb") as file:
        file.write(response.content)

    print(f"Dataset downloaded and extracted to '{data_dir}'.")

    # --------------------------------------------------------
    # Flower names
    # --------------------------------------------------------

    labels_description = [
        'pink primrose',
        'hard-leaved pocket orchid',
        'canterbury bells',
        'sweet pea',
        'english marigold',
        'tiger lily',
        'moon orchid',
        'bird of paradise',
        'monkshood',
        'globe thistle',
        'snapdragon',
        "colt's foot",
        'king protea',
        'spear thistle',
        'yellow iris',
        'globe-flower',
        'purple coneflower',
        'peruvian lily',
        'balloon flower',
        'giant white arum lily',
        'fire lily',
        'pincushion flower',
        'fritillary',
        'red ginger',
        'grape hyacinth',
        'corn poppy',
        'prince of wales feathers',
        'stemless gentian',
        'artichoke',
        'sweet william',
        'carnation',
        'garden phlox',
        'love in the mist',
        'mexican aster',
        'alpine sea holly',
        'ruby-lipped cattleya',
        'cape flower',
        'great masterwort',
        'siam tulip',
        'lenten rose',
        'barbeton daisy',
        'daffodil',
        'sword lily',
        'poinsettia',
        'bolero deep blue',
        'wallflower',
        'marigold',
        'buttercup',
        'oxeye daisy',
        'common dandelion',
        'petunia',
        'wild pansy',
        'primula',
        'sunflower',
        'pelargonium',
        'bishop of llandaff',
        'gaura',
        'geranium',
        'orange dahlia',
        'pink-yellow dahlia?',
        'cautleya spicata',
        'japanese anemone',
        'black-eyed susan',
        'silverbush',
        'californian poppy',
        'osteospermum',
        'spring crocus',
        'bearded iris',
        'windflower',
        'tree poppy',
        'gazania',
        'azalea',
        'water lily',
        'rose',
        'thorn apple',
        'morning glory',
        'passion flower',
        'lotus',
        'toad lily',
        'anthurium',
        'frangipani',
        'clematis',
        'hibiscus',
        'columbine',
        'desert-rose',
        'tree mallow',
        'magnolia',
        'cyclamen',
        'watercress',
        'canna lily',
        'hippeastrum',
        'bee balm',
        'ball moss',
        'foxglove',
        'bougainvillea',
        'camellia',
        'mallow',
        'mexican petunia',
        'bromelia',
        'blanket flower',
        'trumpet creeper',
        'blackberry lily'
    ]

    description_path = os.path.join(
        data_dir,
        "labels_description.txt"
    )

    with open(description_path, "w") as f:

        for label in labels_description:
            f.write(f"{label}\n")


# ============================================================
# 2. CUSTOM DATASET
# ============================================================

class FlowerDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):

        super().__init__()

        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(
            self.root_dir,
            "jpg"
        )

        self.labels = self.load_and_correct_labels()

    def load_and_correct_labels(self):

        labels_mat = scipy.io.loadmat(
            os.path.join(
                self.root_dir,
                "imagelabels.mat"
            )
        )

        # Original labels are 1-102.
        # PyTorch CrossEntropyLoss expects 0-101.
        labels = labels_mat["labels"][0] - 1

        return labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        image = self.retrieve_image(idx)

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])

        return image, label

    def retrieve_image(self, idx):

        img_name = f"image_{idx + 1:05d}.jpg"

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        with Image.open(img_path) as img:

            image = img.convert("RGB")

        return image

    def get_label_description(self, label):

        path_labels_description = os.path.join(
            self.root_dir,
            "labels_description.txt"
        )

        with open(path_labels_description, "r") as f:

            lines = f.readlines()

        return lines[label].strip()


# ============================================================
# 3. DATASET WRAPPER
# ============================================================

class SubsetWithTransform(torch.utils.data.Dataset):

    def __init__(self, subset, transform=None):

        self.subset = subset
        self.transform = transform

    def __len__(self):

        return len(self.subset)

    def __getitem__(self, idx):

        image, label = self.subset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Download dataset
    # --------------------------------------------------------

    download_dataset()

    path_dataset = "Flower_Detection/flower_data"

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # --------------------------------------------------------
    # Train transformation
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.RandomCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    # --------------------------------------------------------
    # Validation/Test transformation
    # --------------------------------------------------------

    val_test_transform = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    # --------------------------------------------------------
    # Create base dataset
    # --------------------------------------------------------

    dataset = FlowerDataset(
        path_dataset,
        transform=None
    )

    print("\nTotal images:", len(dataset))

    # --------------------------------------------------------
    # Train / Validation / Test split
    # --------------------------------------------------------

    total_size = len(dataset)

    train_size = int(0.70 * total_size)

    val_size = int(0.15 * total_size)

    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_subset, val_subset, test_subset = random_split(

        dataset,

        [
            train_size,
            val_size,
            test_size
        ],

        generator=generator
    )

    print("Train:", len(train_subset))
    print("Validation:", len(val_subset))
    print("Test:", len(test_subset))

    # --------------------------------------------------------
    # Apply transforms
    # --------------------------------------------------------

    train_dataset = SubsetWithTransform(
        train_subset,
        transform=train_transform
    )

    val_dataset = SubsetWithTransform(
        val_subset,
        transform=val_test_transform
    )

    test_dataset = SubsetWithTransform(
        test_subset,
        transform=val_test_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    #
    # num_workers=0 is intentional for Windows.
    # --------------------------------------------------------

    batch_size = 32

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True
    )

    # --------------------------------------------------------
    # Check DataLoader
    # --------------------------------------------------------

    images, labels = next(iter(train_loader))

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    # Expected:
    #
    # Image batch shape: torch.Size([32, 3, 224, 224])
    # Label batch shape: torch.Size([32])

    # ========================================================
    # 5. MLP MODEL
    # ========================================================

    class Model(nn.Module):

        def __init__(self):

            super().__init__()

            self.model = nn.Sequential(

                # 3 x 224 x 224
                # = 150528 features

                nn.Flatten(),

                nn.Linear(
                    3 * 224 * 224,
                    1024
                ),

                nn.ReLU(),

                nn.Dropout(0.3),

                nn.Linear(
                    1024,
                    512
                ),

                nn.ReLU(),

                nn.Dropout(0.3),

                nn.Linear(
                    512,
                    256
                ),

                nn.ReLU(),

                # 102 flower classes

                nn.Linear(
                    256,
                    102
                )
            )

        def forward(self, x):

            return self.model(x)

    # ========================================================
    # 6. DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # 7. MODEL
    # ========================================================

    model = Model().to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    # ========================================================
    # 8. LOSS AND OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        model.parameters(),

        lr=0.001
    )

    # ========================================================
    # 9. TRAINING
    # ========================================================

    num_epochs = 10

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0

        train_correct = 0

        train_total = 0

        train_progress = tqdm(

            train_loader,

            desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for images, labels in train_progress:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Clear gradients

            optimizer.zero_grad()

            # Forward pass

            outputs = model(images)

            # Calculate loss

            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation

            loss.backward()

            # Update weights

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            train_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            train_total += labels.size(0)

            train_correct += (
                predictions == labels
            ).sum().item()

        train_accuracy = (
            100.0
            * train_correct
            / train_total
        )

        train_loss /= len(train_loader)

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        val_correct = 0

        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                predictions = outputs.argmax(
                    dim=1
                )

                val_total += labels.size(0)

                val_correct += (
                    predictions == labels
                ).sum().item()

        val_accuracy = (
            100.0
            * val_correct
            / val_total
        )

        val_loss /= len(val_loader)

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(

                {
                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "epoch": epoch + 1,

                    "val_accuracy": val_accuracy
                },

                "best_flower_mlp.pth"
            )

            print(
                "Best model saved!"
            )

    # ========================================================
    # 10. LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        "best_flower_mlp.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"\nLoaded best model from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    # ========================================================
    # 11. TEST
    # ========================================================

    model.eval()

    test_correct = 0

    test_total = 0

    test_loss = 0.0

    with torch.no_grad():

        for images, labels in tqdm(
            test_loader,
            desc="Testing"
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            test_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            test_total += labels.size(0)

            test_correct += (
                predictions == labels
            ).sum().item()

    test_accuracy = (
        100.0
        * test_correct
        / test_total
    )

    test_loss /= len(test_loader)

    # ========================================================
    # 12. FINAL RESULT
    # ========================================================

    print("\n" + "=" * 50)

    print("FINAL TEST RESULTS")

    print("=" * 50)

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    print("=" * 50)


# ============================================================
# IMPORTANT FOR WINDOWS
# ============================================================

if __name__ == "__main__":

    main()