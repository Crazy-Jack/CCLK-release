'''Train an encoder using Contrastive Learning.'''
import argparse
import os
import subprocess
import sys
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from lars import LARS
from tqdm import tqdm

from configs import get_datasets
from critic import LinearCritic
from evaluate import save_checkpoint, encode_train_set, train_clf, test, encode_conditional_train_set, test_conditional
from models import *
from scheduler import CosineAnnealingWithLinearRampLR
from ConditionalSampling import ConditionalSamplingLoss
from threshold_annealing import thresholdAnnealing
from logger import txt_logger

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--base-lr', default=0.25, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint with this filename')
parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset')
parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=100, help='Number of training epochs')
parser.add_argument("--cosine-anneal", action='store_true', help="Use cosine annealing on the learning rate")
parser.add_argument("--no_color_distor", action='store_true', help="Use color distoration or not")
parser.add_argument("--arch", type=str, default='resnet50', help='Encoder architecture')
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
parser.add_argument("--condition_mode", type=str, default='hardnegatives', help='conditional mode, choose from hardnegatives, cl-infonce, or weac-infonce')
parser.add_argument("--lambda_", type=float, default=0.01, help='')
parser.add_argument("--temp_z", type=float, default=1, help='')
parser.add_argument("--scale_z", type=float, default=1, help='')
parser.add_argument("--kz_warmup_epoch", type=int, default=0, help='special case for Kz')
parser.add_argument("--warmup_percent", type=float, default=0.33, help='warmup percent')
parser.add_argument("--start_high_threshold", type=float, default=1., help='annealing')
parser.add_argument("--end_high_threshold", type=float, default=0.6, help='annealing')
parser.add_argument("--start_low_threshold", type=float, default=0., help='annealing')
parser.add_argument("--end_low_threshold", type=float, default=0.4, help='annealing')
parser.add_argument("--save_path", type=str, default="train_related", help="save root folder that will store the results")
parser.add_argument("--customized_name", type=str, default="")
parser.add_argument("--inverse_device", type=str, default='cpu', help="device that compute the inverse")
parser.add_argument("--z_head", type=str, default="", help="specify z head to learn a better representation to condition on")
parser.add_argument("--save_freq", type=int, default=25)
parser.add_argument("--reg_weight", type=float, default=1e-6)
parser.add_argument("--weight_clip_threshold", type=float, default=1e-6, help="clip for computing the weight")
parser.add_argument("--distance_mode", type=str, default="dotProduct")


args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

# folder name
args.save_folder = f"{args.save_path}/{args.condition_mode}/{args.dataset}/bz_{args.batch_size}_ep_{args.num_epochs}"

args.model_name = f"{args.arch}_lr_{args.base_lr}_cosineaneal_{args.cosine_anneal}_lambda_{args.lambda_}"

if args.no_color_distor:
    args.model_name = f"{args.model_name}_nocolordistoration"

args.model_name = f"{args.model_name}_distancemode_{args.distance_mode}"
args.model_name = f"{args.model_name}_inveDevi_{args.inverse_device}_{args.customized_name}"

args.save_location = f"{args.save_folder}/{args.model_name}"
os.makedirs(args.save_location, exist_ok=True)
# args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
# args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_mse = 100000000000  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None

# logger
print('===> Preparing Logger...')
scalar_logger = txt_logger(args.save_location, args, 'python ' + ' '.join(sys.argv))

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset, no_color_distor=args.no_color_distor)

if args.dataset == 'colorMNIST':
    testset, test_conditional_set = testset
    clftrainset, clf_conditional_trainset = clftrainset
else:
    raise NotImplementedError

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
clf_conditional_trainloader = torch.utils.data.DataLoader(clf_conditional_trainset, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)

test_conditional_loader = torch.utils.data.DataLoader(test_conditional_set, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)


# Model
print('==> Building model..')
##############################################################
# Encoder
##############################################################
if args.arch == 'resnet18':
    net = ResNet18(stem=stem)
elif args.arch == 'resnet34':
    net = ResNet34(stem=stem)
elif args.arch == 'resnet50':
    net = ResNet50(stem=stem)
elif args.arch == 'LeNet':
    net = LeNet()
else:
    raise ValueError("Bad architecture specification")
net = net.to(device)

##############################################################
# Critic
##############################################################
critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)
if args.z_head:
    z_head = nn.Sequential(
        nn.Linear(net.representation_dim, net.representation_dim),
        nn.ReLU(),
        nn.Linear(net.representation_dim, net.representation_dim),
        nn.ReLU(),
        nn.Linear(net.representation_dim, int(args.z_head)),
        nn.BatchNorm1d(int(args.z_head), affine=False)
    ).to(device)
else:
    z_head = None

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

encoder_optimizer = LARS(list(net.parameters()) + list(critic.parameters()), lr=args.lr, eta=1e-3, momentum=args.momentum, weight_decay=1e-4, max_epoch=200)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', args.resume)
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint['net'])
    critic.load_state_dict(checkpoint['critic'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    encoder_optimizer.load_state_dict(checkpoint['optim'])

criterion = ConditionalSamplingLoss(mode=args.condition_mode, lambda_=args.lambda_, temp_z=args.temp_z, scale=args.scale_z,
                                    inverse_device=args.inverse_device, weight_clip_threshold=args.weight_clip_threshold, distance_mode=args.distance_mode)

if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(encoder_optimizer, args.num_epochs)

# Training
def train(epoch, args, high_threshold, low_threshold):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0

    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, condition) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        condition = condition.to(device)
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        if z_head is not None:
            z1 = z2 = z_head(condition)
        else:
            z1 = z2 = condition
        loss = criterion(raw_scores, condition1=z1, condition2=z2, high_threshold=high_threshold, low_threshold=low_threshold)

        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


for epoch in range(start_epoch, start_epoch + args.num_epochs):

    high_threshold, low_threshold = thresholdAnnealing(epoch, args)

    loss = train(epoch, args, high_threshold, low_threshold)

    scalar_logger.log_value(epoch, ('loss', loss),
                                    ('high_threshold', high_threshold),
                                    ('low_threshold', low_threshold),
                                    ('learning_rate', encoder_optimizer.param_groups[0]['lr'])
                                    )

    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=args.reg_weight)
        acc = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
        scalar_logger.log_value(epoch, ('Best acc', best_acc),
                                       ('acc', acc))
        del X, y


        X, y = encode_conditional_train_set(clf_conditional_trainloader, device, net)
        clf_conditional = train_clf(X, y, net.representation_dim, 3, device, reg_weight=args.reg_weight, continuous=True)
        mse_loss = test_conditional(test_conditional_loader, device, net, clf_conditional)
        if mse_loss < best_mse:
            best_mse = mse_loss
        scalar_logger.log_value(epoch, ('Best MSE', best_mse),
                                       ('MES', mse_loss))
        del X, y




    if (epoch % args.save_freq == 0) or (epoch == start_epoch + args.num_epochs - 1):
        if (epoch > 0):
            save_checkpoint(net, clf, critic, epoch, args, best_acc, scalar_logger, os.path.basename(__file__))
    if args.cosine_anneal:
        scheduler.step()
