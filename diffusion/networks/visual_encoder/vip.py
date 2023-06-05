

from vip import load_vip


def load_vip():
    # load the VIP model
    model = load_vip()
    model.eval()
    return model
