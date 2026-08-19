'''
The MIT License
Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

References:
https://github.com/PRBonn/lidar-bonnetal
https://github.com/TiagoCortinhal/SalsaNext
'''

import torch.optim.lr_scheduler as toptim


class WarmupCosineLR(toptim._LRScheduler):
    ''' Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    '''

    def __init__(self, optimizer, lr, warmup_steps, momentum, max_steps, min_lr=0.0):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.cosine_scheduler = toptim.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=self.min_lr)

        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        self.last_epoch = -1
        self.finished = False
        super().__init__(optimizer)

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return self.cosine_scheduler.step(epoch)
        else:
            return self.initial_scheduler.step(epoch)

    def state_dict(self):
        state = {
            'lr': self.lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'momentum': self.momentum,
            'max_steps': getattr(self, 'max_steps', None),
            'finished': self.finished,
            'initial_scheduler': self.initial_scheduler.state_dict(),
            'cosine_scheduler': self.cosine_scheduler.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        self.finished = state_dict.get('finished', False)
        if 'initial_scheduler' in state_dict:
            self.initial_scheduler.load_state_dict(state_dict['initial_scheduler'])
        if 'cosine_scheduler' in state_dict:
            self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])

    def fast_forward(self, steps):
        """Fast-forward scheduler steps when resuming without saved scheduler state."""
        for _ in range(steps):
            self.step()


class WarmupLR(toptim._LRScheduler):
    ''' Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    '''

    def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum
        self.decay = decay

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        self.last_epoch = -1
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr * (self.decay ** self.last_epoch) for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return super(WarmupLR, self).step(epoch)
        else:
            return self.initial_scheduler.step(epoch)

    def state_dict(self):
        state = {
            'lr': self.lr,
            'warmup_steps': self.warmup_steps,
            'momentum': self.momentum,
            'decay': self.decay,
            'finished': self.finished,
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            'initial_scheduler': self.initial_scheduler.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        self.finished = state_dict.get('finished', False)
        if 'initial_scheduler' in state_dict:
            self.initial_scheduler.load_state_dict(state_dict['initial_scheduler'])
        if 'base_lrs' in state_dict:
            self.base_lrs = state_dict['base_lrs']
        if 'last_epoch' in state_dict:
            self.last_epoch = state_dict['last_epoch']

    def fast_forward(self, steps):
        """Fast-forward scheduler steps when resuming without saved scheduler state."""
        for _ in range(steps):
            self.step()

