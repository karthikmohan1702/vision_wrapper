from collections import Counter, OrderedDict

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from torchvision import datasets
from tqdm import tqdm


# =============== Data Loader ==============================
class get_cifar10_dataset(torchvision.datasets.CIFAR10):
    def __init__(
        self, root="~/data/cifar10", train=True, download=True, transform=None
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label


def get_mnist_dataset(train_transforms, test_transforms):
    train = datasets.MNIST(
        "./data", train=True, download=True, transform=train_transforms
    )
    test = datasets.MNIST(
        "./data", train=False, download=True, transform=test_transforms
    )
    return train, test


def train_test_loader(trainset, testset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# =============== Augmentations ==============================
def albumentaion_transform(mean: list, std: list):
    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
            A.RandomCrop(width=32, height=32, p=1),
            A.HorizontalFlip(p=0.2),
            A.augmentations.geometric.transforms.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=1,
                fill_value=mean,
                mask_fill_value=None,
            ),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose([A.Normalize(mean, std), ToTensorV2()])

    return train_transform, test_transform


# =============== MISC ==============================
def get_device_info():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    return device


def get_model_summary(model, input_size):
    device = get_device_info()
    model_to_device = model.to(device)
    return summary(model, input_size)


def get_mean_std(loader):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_class_dist(classes, dataset):
    ord_dict = OrderedDict()
    for id, cl in enumerate(classes):
        ord_dict[id] = cl

    count_label = []
    count_img_size = []
    for img, label in dataset:
        label = ord_dict[label]
        count_label.append(label)
        count_img_size.append(img.numpy().shape)
    return Counter(count_label), Counter(count_img_size)


def print_dist_stats(train_set, test_set):
    print("Distribution of classes in Train Dataset")
    print("==" * 30)
    display(get_class_dist(train_set.classes, train_set)[0])
    print("\n")

    print("Distribution of classes in Test Dataset")
    print("==" * 30)
    display(get_class_dist(test_set.classes, test_set)[0])
    print("\n")

    print("Distribution of size of image in Train Dataset")
    print("==" * 30)
    display(get_class_dist(train_set.classes, train_set)[1])
    print("\n")

    print("Distribution of size of image in Test Dataset")
    print("==" * 30)
    display(get_class_dist(test_set.classes, test_set)[1])
    print("\n")


# =============== Grad Cam ==============================
class GradCam:
    def __init__(
        self,
        model,
        img_tensor,
        correct_class,
        classes,
        feature_module,
        target_layer_names,
    ):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.img_tensor = img_tensor.unsqueeze_(0)
        self.classes = classes
        self.correct_class = self.classes[correct_class]
        self.model = model

        target_activations = []
        x = self.img_tensor.to(device)
        for name, module in model._modules.items():
            if module == model.layer4:
                target_activations, x = self.extract_features(
                    x, feature_module, target_layer_names
                )
            elif "linear" in name.lower():
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                x = module(x)
            else:
                x = module(x)

        features, output = target_activations, x
        index = np.argmax(output.cpu().data.numpy())
        self.pred_class = self.classes[index]
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.layer4.zero_grad()
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.img_tensor.shape[2:])
        cam = cam - np.min(cam)
        self.cam = cam / np.max(cam)

    def extract_features(self, input, model, target_layers):
        x = input
        outputs = []
        self.gradients = []
        for name, module in model._modules.items():
            x = module(x)
            if name in target_layers:
                x.register_hook(lambda grad: self.gradients.append(grad))
                outputs += [x]
        return outputs, x

    def plot(self):
        heatmap = cv2.applyColorMap(np.uint8(255 * self.cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img = self.img_tensor[0] / 2 + 0.5
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam = heatmap + img
        cam = cam / np.max(cam)
        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(3, 5, 1)
        ax.set_title("Input Image")
        plt.imshow(img)

        ax = fig.add_subplot(3, 5, 2)
        ax.set_title(f"pred: {self.pred_class}" f" / correct: {self.correct_class}")
        plt.imshow(cam)

        fig.tight_layout()


def plot_grad_cam(test_loader, model, classes, samples: int):
    img_tensor_batch, labels = next(iter(test_loader))
    for idx in range(samples):
        gm = GradCam(
            model=model,
            img_tensor=img_tensor_batch[idx],
            correct_class=labels[idx],
            classes=classes,
            feature_module=model.layer4,
            target_layer_names=["1"],
        )
        gm.plot()


# =============== Visualization ==============================


def plot_sample_img(train_set, test_set, batch_size=5):
    sample_data_loader, _ = train_test_loader(train_set, test_set, batch_size)
    classes = train_set.classes
    print("Classes present in the dataset ==> ", classes)
    print("\n")
    dataiter = iter(sample_data_loader)
    images, labels = dataiter.next()
    grid = torchvision.utils.make_grid(images)
    plt.axis("off")
    print(" ".join("%7s" % classes[labels[j]] for j in range(5)))
    _ = plt.imshow(grid.permute(1, 2, 0))


def plot_acc_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def misclassified_images(model, classes, test_transform, trainset, testset):
    testset = get_cifar10_dataset(train=False, transform=test_transform)
    batch_size = len(testset)
    train_loader, test_loader = train_test_loader(
        trainset, testset, batch_size
    )
    mean, std = get_mean_std(train_loader)
    device = get_device_info()
    model = model.to(device)
    wrong_imgs = []
    wrong_lbl = []
    corr_lbl = []
    with torch.no_grad():
        for test_idx, test_img in enumerate(test_loader):
            data, target = test_img
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            indexes = (
                pred.view(
                    -1,
                )
                != target.view(
                    -1,
                )
            ).nonzero()
            fig = plt.figure(figsize=(10, 9))
            for i, idx in enumerate(indexes[:15]):
                ax = fig.add_subplot(3, 5, i + 1)
                mean_norm = torch.tensor(mean).reshape(1, 3, 1, 1)
                std_norm = torch.tensor(std).reshape(1, 3, 1, 1)
                img = data.cpu() * std_norm + mean_norm
                wrong_imgs.append(img[idx].squeeze().permute(1, 2, 0))
                wrong_lbl.append(classes[pred[idx].item()])
                corr_lbl.append(classes[target[idx].item()])
                ax.imshow(img[idx].squeeze().permute(1, 2, 0))
                # ax.imshow(img[idx].squeeze().permute(1, 2, 0).clamp(0, 1))
                ax.set_title(
                    f"Target = {classes[target[idx].item()]} \n Predicted = {classes[pred[idx].item()]}"
                )

            plt.show()

