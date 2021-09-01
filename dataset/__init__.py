from .base import Set

def load(name, root, dpath, mode, transform = None):
    return Set(root = root, dpath = dpath, mode = mode, transform = transform)
    
