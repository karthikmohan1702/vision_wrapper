import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import torch.optim as optim
from torch_lr_finder import LRFinder


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

def lr_finder(param_dict):
    """
    Learning rate finder takes the params in and increases the learning rate
    in between the boundaries mentioned in the params & provides info the range of 
    learning rates that can be used.

    Parameters
    ==========
    param_dict: dict - Has all the params required for LRFinder
    
    Returns
    =======
    lr_max: int - maximum lr value
    lr_finder: dict - has history of loss & lr
    """
    model = param_dict["model"]
    train_loader = param_dict["train_loader"]
    test_loader = param_dict["test_loader"]
    criterion = param_dict["criterion"]
    optimizer = param_dict["optimizer"]
    end_lr = param_dict["end_lr"]
    num_iter = param_dict["num_iter"]
    step_mode = param_dict["step_mode"]
    device = param_dict["device"]

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, test_loader, end_lr, num_iter, step_mode)
    lr_max = lr_finder.history['lr']
    lr_max = lr_max[lr_finder.history['loss'].index(lr_finder.best_loss)]
    return lr_max, lr_finder


def run_model(train_loader, test_loader, model, epochs, device,  max_at_epoch, param_dict):
    """
    Training the model by defined optimizer, scheduler & epochs and 
    using the lr_finder, will arrive at the lr_max. 
    """
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    # Calling LR Finder & getting the lr_max
    lr_max, _ = lr_finder(param_dict)

    # Getting lr_min from the param_dict
    lr_min = param_dict["lr_min"]

    EPOCHS = epochs
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    
    # Schedule the scheduler
    scheduler = OneCycleLR(
        optimizer, max_lr=lr_max, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=max_at_epoch/EPOCHS
    )

    # Running the epochs
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        scheduler.step()
        test(model, device, test_loader, test_acc, test_losses)

    return train_losses, test_losses, train_acc, test_acc
