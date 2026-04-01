import os
import copy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchvision
import torchvision.models as models

import util


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ckpt.pth.tar')


class Net(nn.Module):
    def __init__(self, image_size):
        super(Net, self).__init__()
        if image_size == 256:
            linear_input_size_fc1 = 64 * 9 * 9
        elif image_size == 224:
            linear_input_size_fc1 = 64 * 8 * 8
        else:
            raise ValueError('Unsupported image size: {}'.format(image_size))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(linear_input_size_fc1, 256),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze(-1)


def train(epoch, net, criterion, optimizer, trainLoader, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        util.progress_bar(batch_idx, len(trainLoader),
                          'Train Loss: %.3f' % (train_loss / (batch_idx + 1)))


def test(epoch, net, criterion, testLoader, device):
    net.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            util.progress_bar(batch_idx, len(testLoader),
                              'Test Loss: %.3f' % (test_loss / (batch_idx + 1)))

    return test_loss / len(testLoader)


def save_checkpoint(net, optimizer, scheduler, loss, epoch, checkpoint_path=CHECKPOINT_PATH):
    print("==> Saving checkpoint..")
    state = {
        'model_state_dict': net.state_dict(),
        'best_loss': loss,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'rng_state': torch.get_rng_state(),
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


def load_checkpoint(net, optimizer, scheduler=None, checkpoint_path=CHECKPOINT_PATH, map_location='cpu'):
    print('==> Resuming from checkpoint..')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('Error: No checkpoint file found at {}'.format(checkpoint_path))

    device = torch.device(map_location)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    move_optimizer_state_to_device(optimizer, device)

    scheduler_state_dict = checkpoint.get('scheduler_state_dict')
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    rng_state = checkpoint.get('rng_state')
    if rng_state is not None:
        torch.set_rng_state(rng_state)

    best_loss = checkpoint.get('best_loss', float('inf'))
    start_epoch = checkpoint.get('epoch', -1) + 1
    return best_loss, start_epoch


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def compute_metrics(net, X_test, y_test, device):
    net.eval()
    with torch.no_grad():
        y_pred = net(X_test.to(device))
    y_pred = y_pred.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2
    }
    return metrics


def build_model_and_optimizer(args, device):
    if args.arch is None:
        image_size = 224 if args.augment else 256
        net = Net(image_size).to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
        return net, optimizer

    if args.arch == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1), nn.ReLU())
        net = net.to(device)
        optimizer = optim.Adam(net.classifier[6].parameters(),
                               lr=args.lr, weight_decay=args.decay)
        return net, optimizer

    resnets = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    net = resnets[args.arch](pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.ReLU())
    net = net.to(device)
    optimizer = optim.Adam(net.fc.parameters(), lr=args.lr, weight_decay=args.decay)
    return net, optimizer


def perform_cnn(dataset, args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    best_loss = float('inf')

    X_train, X_test, y_train, y_test = dataset
    trainSet = utils.TensorDataset(X_train, y_train)
    testSet = utils.TensorDataset(X_test, y_test)
    trainLoader = utils.DataLoader(trainSet, shuffle=True,
                                   batch_size=args.batch_size)
    testLoader = utils.DataLoader(testSet, shuffle=False,
                                  batch_size=args.batch_size)

    if args.metric == 'mean_squared_error':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    net, optimizer = build_model_and_optimizer(args, device)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                             step_size=10)
    start_epoch = args.start_epoch
    best_state_dict = copy.deepcopy(net.state_dict())

    if args.resume:
        best_loss, start_epoch = load_checkpoint(
            net,
            optimizer,
            scheduler=lr_scheduler,
            map_location=device,
        )
        best_state_dict = copy.deepcopy(net.state_dict())

    for epoch in range(start_epoch, args.epochs):
        train(epoch, net, criterion, optimizer, trainLoader, device)
        lr_scheduler.step()
        test_loss = test(epoch, net, criterion, testLoader, device)
        if test_loss < best_loss:
            best_loss = test_loss
            best_state_dict = copy.deepcopy(net.state_dict())
            save_checkpoint(net, optimizer, lr_scheduler, best_loss, epoch)

    net.load_state_dict(best_state_dict)

    return compute_metrics(net, X_test, y_test, device)
