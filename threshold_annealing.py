



def thresholdAnnealing(epoch, opt):
    """annealing high and lower threshold based on the process of training
    using the warmup phase for annealing and steady after training
    """

    if not hasattr(opt, 'kz_warmup_epoch'):
        opt.kz_warmup_epoch = 0

    if epoch < opt.kz_warmup_epoch:
        progress = 0.0
    else:
        assert opt.num_epochs * opt.warmup_percent != 0
        progress = min(epoch / (opt.num_epochs * opt.warmup_percent), 1.)


    assert opt.start_high_threshold >= opt.end_high_threshold
    assert opt.start_low_threshold <= opt.end_low_threshold
    assert opt.start_high_threshold >= opt.start_low_threshold
    assert opt.end_high_threshold >= opt.end_low_threshold

    high_threshold =  opt.start_high_threshold - progress * (opt.start_high_threshold - opt.end_high_threshold)
    low_threshold = opt.start_low_threshold + progress * (opt.end_low_threshold - opt.start_low_threshold)

    return high_threshold, low_threshold
