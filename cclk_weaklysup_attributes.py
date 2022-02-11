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
from evaluate import save_checkpoint, encode_train_set, train_clf, test
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
parser.add_argument("--arch", type=str, default='resnet50', help='Encoder architecture',
                    choices=['resnet18', 'resnet34', 'resnet50'])
parser.add_argument("--num-workers", type=int, default=8, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
parser.add_argument("--condition_mode", type=str, default='cl-infonce', help='conditional mode, choose from hardnegatives, cl-infonce, or weac-infonce')
parser.add_argument("--lambda_", type=float, default=0.01, help='')
parser.add_argument("--temp_z", type=float, default=1, help='')
parser.add_argument("--scale_z", type=float, default=1, help='')
parser.add_argument("--kz_warmup_epoch", type=int, default=0, help='special case for Kz')
parser.add_argument("--update_condition_eps", type=int, default=1, help='update epoch for conditions')
parser.add_argument("--kz_momentum", type=float, default=0.9, help="momentum for making the feature conditioned on more stable, 1 means no momentum, f=m*current + (1-m)*f")
parser.add_argument("--warmup_percent", type=float, default=0.33, help='warmup percent')
parser.add_argument("--start_high_threshold", type=float, default=1., help='annealing')
parser.add_argument("--end_high_threshold", type=float, default=0.6, help='annealing')
parser.add_argument("--start_low_threshold", type=float, default=0., help='annealing')
parser.add_argument("--end_low_threshold", type=float, default=0.4, help='annealing')
parser.add_argument("--weight_clip_threshold", type=float, default=1e-6, help="clip for computing the weight")
parser.add_argument("--save_path", type=str, default="train_related", help="save root folder that will store the results")
parser.add_argument("--distance_mode", type=str, default='dotProduct', help='distance mode for conditioning')
parser.add_argument("--customized_name", type=str, default="")
parser.add_argument("--num_attr", type=int, default=-1)
parser.add_argument("--inverse_device", type=str, default='cpu', help="device that compute the inverse")
parser.add_argument("--inverse_gradient", action='store_true', help="indicate whether the inverse has gradient")
parser.add_argument("--noinverse", action='store_true', help="indicate whether or not there is inverse added, default is having inverse")
parser.add_argument("--condition_head", type=str, default="", help="architectures for head, seperated by -, e.g. 2048-2048-128")
parser.add_argument("--em_conditional_head", action='store_true', help="indicate whether to use EM-like way to find the right mlp-head encoder")
parser.add_argument("--clf_reg_weight_decay", type=float, default=1e-5)
parser.add_argument("--save_freq", type=int, default=50)


args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

# folder name
args.save_folder = f"{args.save_path}/{args.condition_mode}-attributes/{args.dataset}/bz_{args.batch_size}_ep_{args.num_epochs}"

args.model_name = f"{args.arch}_lr_{args.base_lr}_cosineaneal_{args.cosine_anneal}_lambda_{args.lambda_}_scale_{args.scale_z}"


args.model_name = f"{args.model_name}_weight_clip_threshold_{args.weight_clip_threshold}"
args.model_name = f"{args.model_name}_kzwarmupep_{args.kz_warmup_epoch}"
args.model_name = f"{args.model_name}_distanceMode_{args.distance_mode}"
args.model_name = f"{args.model_name}_num_attr_{args.num_attr}"
args.model_name = f"{args.model_name}_lambda_{args.lambda_}"

args.model_name = f"{args.model_name}_tempz{args.temp_z}"

if args.condition_head:
    args.model_name = f"{args.model_name}_conditionHead_{args.condition_head}"

if args.noinverse:
    args.model_name = f"{args.model_name}_noinverse"
if args.inverse_gradient:
    args.model_name = f"{args.model_name}_inverseGradient"

args.model_name = f"{args.model_name}_regweightdecay_{args.clf_reg_weight_decay}"
args.model_name = f"{args.model_name}_{args.customized_name}"



args.save_location = f"{args.save_folder}/{args.model_name}"
os.makedirs(args.save_location, exist_ok=True)
# args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
# args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None

# logger
print('===> Preparing Logger...')
scalar_logger = txt_logger(args.save_location, args, 'python ' + ' '.join(sys.argv))

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset,
                                                    attributes=True, num_attr=args.num_attr)





trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

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
else:
    raise ValueError("Bad architecture specification")
net = net.to(device)
condition_head = None
condition_head_optimizer = None
if args.condition_head:
    num_attr = len(trainset.attr_list) if args.num_attr == -1 else args.num_attr
    archit = [num_attr] + [int(i) for i in args.condition_head.split("-")]
    modules = []
    for i in range(len(archit)-1):
        modules.extend([
            nn.Linear(archit[i], archit[i+1]),
            nn.ReLU(),
        ])
    modules.append(nn.Sigmoid())
    condition_head = nn.Sequential(*modules).to(device)
    condition_head_optimizer = LARS(condition_head.parameters(), lr=args.lr, eta=1e-3, momentum=args.momentum, weight_decay=1e-6)

##############################################################
# Critic
##############################################################
critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    critic.load_state_dict(checkpoint['critic'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = ConditionalSamplingLoss(mode=args.condition_mode, lambda_=args.lambda_,temp_z=args.temp_z, scale=args.scale_z, weight_clip_threshold=args.weight_clip_threshold,
                                    distance_mode=args.distance_mode,
                                    inverse_gradient=args.inverse_gradient)
encoder_optimizer = LARS(list(net.parameters()) + list(critic.parameters()), lr=args.lr, eta=1e-3, momentum=args.momentum, weight_decay=1e-6, max_epoch=200)

if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(encoder_optimizer, args.num_epochs)

# Training
def train(epoch, args, high_threshold, low_threshold):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0

    condition_batch_origin = torch.eye(args.batch_size).to(device)
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, attributes, index) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        bz = x1.shape[0]
        encoder_optimizer.zero_grad()
        if condition_head:
            condition_head_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        # add condition input

        # if isinstance(condition_encode, torch.Tensor):
        #     condition_batch = condition_encode[index] # get condition if it's not None
        # else:
        #     if bz != args.batch_size:
        #         condition_batch = condition_batch_origin[:bz, :bz]
        #     else:
        #         condition_batch = condition_batch_origin

        condition_batch = attributes.to(device)
        if condition_head:
            condition_batch = condition_head(condition_batch)

        assert isinstance(condition_batch, torch.Tensor), "condition on None Tensor type"
        warmup = True if epoch < args.kz_warmup_epoch else False
        loss = criterion(raw_scores,
                         condition1=condition_batch,
                         condition2=condition_batch,
                         high_threshold=high_threshold, low_threshold=low_threshold,
                         warmup=warmup)
        loss.backward()
        encoder_optimizer.step()
        if condition_head:
            condition_head_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


for epoch in range(start_epoch, args.num_epochs):
    
    high_threshold, low_threshold = None, None
    # encode the representation for conditioning

    loss = train(epoch, args, high_threshold, low_threshold)


    if args.kernel_mode not in ['self']:
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('high_threshold', high_threshold),
                                    ('low_threshold', low_threshold),
                                    ('learning_rate', encoder_optimizer.param_groups[0]['lr'])
                                    )
    else:
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('learning_rate', encoder_optimizer.param_groups[0]['lr'])
                                    )

    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=args.clf_reg_weight_decay)
        acc = test(testloader, device, net, clf)


        if acc > best_acc:
            best_acc = acc
        scalar_logger.log_value(epoch, ('Best Acc', best_acc))

    if (clf is not None):
        if (epoch % args.save_freq == 0) or ((epoch + 1) == args.num_epochs):
            save_checkpoint(net, clf, critic, epoch, args, best_acc, scalar_logger, os.path.basename(__file__), encoder_optimizer)
    if args.cosine_anneal:
        scheduler.step()
