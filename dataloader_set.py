from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cifar10_classes =["airplane", "automobile", "bird",
                  "cat", "deer", "dog", "frog", "horse",
                  "ship", "truck"]

affine = transforms.RandomAffine((-30, 30),
                                 scale=(0.8, 1.2))
flip = transforms.RandomHorizontalFlip(p=0.5)
normalize = transforms.Normalize((0.0, 0.0, 0.0),
                                 (1.0, 1.0, 1.0))
to_tensor = transforms.ToTensor()

transform_train = transforms.Compose([affine,
                                      flip, to_tensor, normalize])
transform_test = transforms.Compose([to_tensor, normalize])

cifar10_train = CIFAR10("./data", train=True,
                      download=True, transform=transform_train)
cifar10_test = CIFAR10("./data", train=False,
                       download=True, transform=transform_test)

batch_size = 64
train_loader = DataLoader(cifar10_train,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test,
                         batch_size=batch_size, shuffle=False)
