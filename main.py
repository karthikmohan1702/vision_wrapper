import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.optim as optim
import utils


def l1_regularization(model, loss, lambda_l1):
    l1 = 0
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    loss = loss + lambda_l1 * l1
    return loss


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    train_losses,
    train_acc,
    lambda_l1=0,
):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(y_pred, target)

        # L1 regularization
        if lambda_l1 > 0:
            loss = l1_regularization(loss, lambda_l1)

        train_losses.append(loss.data.cpu().numpy().item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, test_acc, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_acc.append(100.0 * correct / len(test_loader.dataset))


def run_model(train_loader, test_loader, model, epochs):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    EPOCHS = epochs

    device = utils.get_device_info()

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = OneCycleLR(
        optimizer, max_lr=0.05, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    model.eval()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        scheduler.step()
        test(model, device, test_loader, test_acc, test_losses)

    return train_losses, test_losses, train_acc, test_acc
