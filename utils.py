# utils.py
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0

class EarlyStopping:
    """Stop if metric hasn't improved for `patience` epochs"""
    def __init__(self, patience=8, minimize=True):
        self.patience = patience
        self.best = float("inf") if minimize else -float("inf")
        self.counter = 0
        self.minimize = minimize
    def __call__(self, value):
        improved = (value < self.best) if self.minimize else (value > self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
