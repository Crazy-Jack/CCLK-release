import os

import torch


def save_checkpoint(net, clf, critic, epoch, args, acc, scalar_logger, script_name, optim):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'script': script_name,
        'acc': acc,
        'optim': optim.state_dict(),
    }

    scalar_logger.log_value(epoch, ('Acc', acc))

    if not os.path.isdir(args.save_location):
        os.makedirs(args.save_location, exist_ok=True)
    destination = os.path.join(args.save_location, f"ckpt_epoch_{epoch}.pth")
    torch.save(state, destination)
