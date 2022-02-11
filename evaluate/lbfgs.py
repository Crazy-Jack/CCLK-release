import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def encode_train_set(clftrainloader, device, net):
    net.eval()
    store = []
    with torch.no_grad():
        t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            store.append((representation, targets))

            t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y

def encode_conditional_train_set(clf_conditional_trainloader, device, net):
    net.eval()
    store = []
    with torch.no_grad():
        t = tqdm(enumerate(clf_conditional_trainloader), desc='Encoded: **/** ', total=len(clf_conditional_trainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, _, conditions) in t:
            inputs, conditions = inputs.to(device), conditions.to(device)
            representation = net(inputs)
            store.append((representation, conditions))
            t.set_description('Encoded %d/%d' % (batch_idx, len(clf_conditional_trainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y

def train_clf(X, y, representation_dim, num_classes, device, reg_weight=1e-3, optim_choice='lbfgs', continuous=False):
    print('\nL2 Regularization weight: %g' % reg_weight)
    print(f"optim_choice {optim_choice}")

    if continuous:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()


    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    if optim_choice == 'lbfgs':
        n_optim_steps = 250 * 2
        clf_optimizer = optim.LBFGS(clf.parameters())
    elif optim_choice == 'adam':
        n_optim_steps = 250 * 20
        clf_optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=reg_weight)
    else:
        raise NotImplementedError

    clf.train()

    t = tqdm(range(n_optim_steps), desc='Loss: **** | Train Acc: ****% ', bar_format='{desc}{bar}{r_bar}')
    for _ in t:
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X)
            loss = criterion(raw_scores, y)
            if continuous:
                loss_mse = loss
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()

            if not continuous:
                _, predicted = raw_scores.max(1)
                correct = predicted.eq(y).sum().item()

                t.set_description('Loss: %.3f | Train Acc: %.3f%% ' % (loss, 100. * correct / y.shape[0]))
            else:
                t.set_description('Loss: %.10f%% ' % loss_mse)

            return loss

        clf_optimizer.step(closure)

    return clf


def test(testloader, device, net, clf):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, targets)

            test_clf_loss += clf_loss.item()
            _, predicted = raw_scores.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    return acc



def test_conditional(testloader, device, net, clf):
    criterion = nn.MSELoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, _, condition) in t:
            inputs, condition = inputs.to(device), condition.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, condition)

            test_clf_loss += clf_loss.item()

            t.set_description('Loss: %.10f%% ' % (test_clf_loss / (batch_idx + 1)))

    return test_clf_loss / (batch_idx + 1)
