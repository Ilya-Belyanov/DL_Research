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


def compute_error_model(model, loader, device):
    # Evaluation mode
    model.eval()

    false_imgs, false_labels, true_labels = [], [], []
    for i, (x, y) in enumerate(loader):
        x_gpu, y_gpu = x.to(device), y.to(device)

        y_probs = model(x_gpu)
        y_hat = torch.argmax(y_probs, 1)

        false_imgs += x_gpu[y_hat != y_gpu].cpu().numpy().tolist()
        false_labels += y_hat[y_hat != y_gpu].cpu().numpy().tolist()
        true_labels += y_gpu[y_hat != y_gpu].cpu().numpy().tolist()

    return false_imgs, false_labels, true_labels
