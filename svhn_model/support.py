import torch
import torch.nn as nn

class Flattener(nn.Module):
    """Слой переводит многомерный тензор в одномерный"""
    def forward(self, x):
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)


def do_epoch(model, train_loader, loss_function, optimizer, device):
    # Enter train mode
    model.train()

    total_loss, correct_samples, total_samples = 0, 0, 0
    for i_step, (x, y) in enumerate(train_loader):
        x_gpu, y_gpu = x.to(device), y.to(device)

        # Prediction
        prediction = model(x_gpu)
        loss = loss_function(prediction, y_gpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, indices = torch.max(prediction, 1)
        correct_samples += torch.sum(indices == y_gpu)
        total_samples += y.shape[0]
        total_loss += float(loss)
    return total_loss / i_step, float(correct_samples) / total_samples


def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None,
                scheduler_loss=False):
    train_losses, train_accuracy_history = [], []
    validate_losses, validate_accuracy_history = [], []

    for epoch in range(num_epochs):

        train_loss, train_accuracy = do_epoch(model, train_loader, loss_function, optimizer, device)
        validate_loss, validate_accuracy = 0, 0
        if scheduler is not None:
            if scheduler_loss:
                scheduler.step(validate_loss)
            else:
                scheduler.step()

        validate_loss, validate_accuracy = compute_loss_accuracy(model, val_loader, loss_function, device)

        train_losses.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        validate_losses.append(validate_loss)
        validate_accuracy_history.append(validate_accuracy)

        print("Epoch #%s - train loss: %f, accuracy: %f | val loss: %f, accuracy: %f" % (
            epoch, train_losses[-1], train_accuracy_history[-1], validate_loss, validate_accuracy))

    return train_losses, train_accuracy_history, validate_losses, validate_accuracy_history


def compute_loss_accuracy(model, loader, loss_function, device):
    # Evaluation mode
    model.eval()

    total_loss, correct, total = 0.0, 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x_gpu, y_gpu = x.to(device), y.to(device)

        y_probs = model(x_gpu)
        y_hat = torch.argmax(y_probs, 1)

        loss = loss_function(y_probs, y_gpu)
        total_loss += float(loss)
        correct += float(torch.sum(y_hat == y_gpu))
        total += y_gpu.shape[0]

    return total_loss / (i + 1), correct / total


def accuracy_number(model, loader, number, device):
    # Evaluation mode
    model.eval()

    correct, total = 0.0, 0.0

    for i, (x, y) in enumerate(loader):
        x_gpu, y_gpu = x.to(device), y.to(device)

        y_number = y_gpu[y_gpu == number]
        x_number = x_gpu[y_gpu == number]

        if len(y_number) == 0:
            continue

        y_probs = model(x_number)
        y_hat = torch.argmax(y_probs, 1)

        correct += float(torch.sum(y_hat == y_number))
        total += x_number.shape[0]

    return correct / total


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.model = nn.Sequential(
            # C1 In 32x32@3, out 32x32@16 - 160 параметров
            nn.Conv2d(3, 16, 3, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            # S2 In 32x32@16, out 16x16@16
            nn.MaxPool2d(kernel_size=2),

            # C3 In 16x16@16, out 14x14@32 - 320 параметров
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            # S4 In 14x14@32, out 7x7@32
            nn.MaxPool2d(kernel_size=2),

            # C5 In 7x7@32, out 5x5@64 - 640 параметров
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            Flattener(),

            # F6 In 5 * 5 * 64, out  - 128.204 + 128 параметров
            nn.Linear(5 * 5 * 64, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),

            # O7 In 128, out 10 - 1280 + 10 параметров
            nn.Linear(128, 10),
          )

  def forward(self, x):
    output = self.model(x)
    return output