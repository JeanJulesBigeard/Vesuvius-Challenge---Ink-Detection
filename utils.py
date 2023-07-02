import numpy as np
import torch
import random
import os


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode="train"):
    set_seed(cfg.seed)
    if mode == "train":
        make_dirs(cfg)


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (
        (1 + beta_squared)
        * (c_precision * c_recall)
        / (beta_squared * c_precision + c_recall + smooth)
    )

    return dice


def calc_fbeta(mask, mask_pred, Logger):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(10, 50 + 1, 5)) / 100:
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f"th: {th}, fbeta: {dice}")

        if dice > best_dice:
            best_dice = dice
            best_th = th

    Logger.info(f"best_th: {best_th}, fbeta: {best_dice}")
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred, Logger):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred, Logger)

    return best_dice, best_th


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def normalization(x):
    """input.shape=(batch,f1,f2,...)"""
    # [batch,f1,f2]->dim[1,2]
    dim = list(range(1, x.ndim))
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + 1e-9)


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def TTA(x, model):
    # x.shape=(batch,c,h,w)
    shape = x.shape
    x = [x, *[torch.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = torch.cat(x, dim=0)
    x = model(x)
    x = torch.sigmoid(x)
    x = x.reshape(4, shape[0], *shape[2:])
    x = [torch.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = torch.stack(x, dim=0)
    return x.mean(0)
