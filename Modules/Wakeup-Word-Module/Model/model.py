import torch
import torch.nn.functional as F
from dataset import *
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


class WakeupWordModel(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Input shape: (batch_size, n_mfcc, time_frames)
        # Add channel dimension: (batch_size, 1, n_mfcc, time_frames)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Third conv block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Adaptive pooling to (batch_size, 128, 1, 1)
        x = self.adaptive_pool(x)

        # Flatten to (batch_size, 128)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


def train_model(dataloader, dataset, optimizer, model, criterion, device):
    size = len(dataset)
    model.train()  # Set mode
    for batch, (features, label) in enumerate(dataloader):
        features = features.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # Reset gradients

        logits = model(features)  # Run model

        loss = criterion(logits.squeeze(1), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(features)  # type: ignore
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


@torch.inference_mode()
def eval_model(dataloader, device, model, best_f1=None):
    model.eval()
    true_labels, pred_labels = [], []

    for features, labels in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features).squeeze(1)
        probability = torch.sigmoid(logits)
        prediction = (probability >= 0.5).int()
        true_labels.extend(labels.cpu().numpy().astype(int))
        pred_labels.extend(prediction.cpu().numpy())

    rep = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )
    print(classification_report(true_labels, pred_labels, zero_division=0))
    print(confusion_matrix(true_labels, pred_labels))

    f1_positive = rep["1"]["f1-score"]  # type: ignore
    RESULTS_0.loc[len(RESULTS_0)] = rep["0"]  # type: ignore
    RESULTS_1.loc[len(RESULTS_1)] = rep["1"]  # type: ignore

    return best_f1, f1_positive


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 64
    THIS_FOLDER = Path(__file__).parent
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_0 = pd.DataFrame(
        {"f1-score": [], "precision": [], "recall": [], "support": []}
    )
    RESULTS_1 = pd.DataFrame(
        {"f1-score": [], "precision": [], "recall": [], "support": []}
    )

    # Lets do it
    WORKING_DIR = Path(f"{THIS_FOLDER}/Datasets")
    TRAIN_DATASET = WakeupWordDataset(WORKING_DIR, "Train")
    TEST_DATASET = WakeupWordDataset(WORKING_DIR, "Test")
    MODEL = WakeupWordModel().to(DEVICE)
    TRAIN_LABELS = TRAIN_DATASET.annos["label"].value_counts(sort=False)
    print(f"Train {TRAIN_LABELS}")
    TEST_LABELS = TEST_DATASET.annos["label"].value_counts(sort=False)
    print(f"Test {TEST_LABELS}")
    CRITERION = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(25740 / 13000))
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=1e-3)
    TRAIN_LOADER = DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    TEST_LOADER = DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    PATIENCE = 10
    MIN_DELTA = 0.001

    best_f1 = None
    epochs_without_improvement = 0
    try:
        for epoch in range(EPOCHS):

            print(f"Epoch {epoch}\n-------------------------------")
            train_model(
                dataloader=TRAIN_LOADER,
                dataset=TRAIN_DATASET,
                optimizer=OPTIMIZER,
                model=MODEL,
                criterion=CRITERION,
                device=DEVICE,
            )
            best_f1, f1_positive = eval_model(
                dataloader=TEST_LOADER, device=DEVICE, model=MODEL, best_f1=best_f1
            )

            if best_f1 is None or f1_positive > best_f1:
                torch.save(MODEL.state_dict(), "best_model.pt")
                print(f"Neues bestes Modell gespeichert (F1={f1_positive:.4f})")
                best_f1 = f1_positive
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"Keine Verbesserung seit {epochs_without_improvement} Epoche(n)")

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early Stopping nach Epoche {epoch} (Patience={PATIENCE} erreicht)"
                )
                break
    except Exception as ex:  # noqa: BLE001
        print(ex)

    RESULTS_0.to_csv("results_0.csv")
    RESULTS_1.to_csv("results_1.csv")
