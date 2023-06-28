import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
from tqdm import tqdm
from utils import init_logger, cfg_init, calc_cv, calc_fbeta
from data.utils import get_train_valid_dataset
from model import build_model
from loss import get_scheduler, AverageMeter, criterion, scheduler_step
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt


def get_transforms(data, cfg):
    if data == "train":
        aug = A.Compose(cfg.train_aug_list)
    elif data == "valid":
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data["image"]
            label = data["mask"]

        return image, label


class CFG:
    # ============== comp exp name =============
    comp_name = "vesuvius"

    comp_dir_path = "data/"
    comp_folder_name = "vesuvius-challenge-ink-detection"
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f"{comp_dir_path}{comp_folder_name}/"

    exp_name = "vesuvius_2d_slide_exp001"

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = "Unet"
    backbone = "efficientnet-b0"
    # backbone = 'se_resnext50_32x4d'

    in_chans = 6  # 65
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 2

    train_batch_size = 16  # 32
    valid_batch_size = train_batch_size * 2
    use_amp = True

    scheduler = "GradualWarmupSchedulerV2"
    # scheduler = 'CosineAnnealingLR'
    epochs = 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 1

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = "maximize"  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = "best"  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 4

    seed = 42

    # ============== set dataset path =============
    print("set dataset path")

    outputs_path = f"working/outputs/{comp_name}/{exp_name}/"

    submission_dir = outputs_path + "submissions/"
    submission_path = submission_dir + f"submission_{exp_name}.csv"

    model_dir = outputs_path + f"{comp_name}-models/"

    figures_dir = outputs_path + "figures/"

    log_dir = outputs_path + "logs/"
    log_path = log_dir + f"{exp_name}.txt"

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_width=int(size * 0.3),
            max_height=int(size * 0.3),
            mask_fill_value=0,
            p=0.5,
        ),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]


def train_fn(train_loader, model, criterion, optimizer, device, train_loss_fc):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels, train_loss_fc)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg


def valid_fn(
    valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, valid_loss_fc
):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(
        enumerate(valid_loader), total=len(valid_loader)
    ):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels, valid_loss_fc)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to("cpu").numpy()
        start_idx = step * CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    print(f"mask_count_min: {mask_count.min()}")
    mask_pred /= mask_count
    return losses.avg, mask_pred


def main():
    cfg_init(CFG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger = init_logger(log_file=CFG.log_path)

    Logger.info("\n-------- exp_info -----------------")
    (
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        valid_xyxys,
    ) = get_train_valid_dataset(CFG)

    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        transform=get_transforms(data="train", cfg=CFG),
    )
    valid_dataset = CustomDataset(
        valid_images,
        CFG,
        labels=valid_masks,
        transform=get_transforms(data="valid", cfg=CFG),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

    DiceLoss = smp.losses.DiceLoss(mode="binary")
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()

    alpha = 0.5
    beta = 1 - alpha
    TverskyLoss = smp.losses.TverskyLoss(
        mode="binary", log_loss=False, alpha=alpha, beta=beta
    )

    fragment_id = CFG.valid_id

    valid_mask_gt = cv2.imread(
        CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0
    )
    valid_mask_gt = valid_mask_gt / 255
    pad0 = CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size
    pad1 = CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

    fold = CFG.valid_id

    if CFG.metric_direction == "minimize":
        best_score = np.inf
    elif CFG.metric_direction == "maximize":
        best_score = -1

    best_loss = np.inf

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, device, BCELoss)

        # eval
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, BCELoss
        )

        scheduler_step(scheduler, epoch)

        best_dice, best_th = calc_cv(valid_mask_gt, mask_pred, Logger)

        # score = avg_val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
        Logger.info(f"Epoch {epoch+1} - avgScore: {score:.4f}")

        if CFG.metric_direction == "minimize":
            update_best = score < best_score
        elif CFG.metric_direction == "maximize":
            update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            Logger.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")

            torch.save(
                {"model": model.state_dict(), "preds": mask_pred},
                CFG.model_dir + f"{CFG.model_name}_fold{fold}_best.pth",
            )

    check_point = torch.load(
        CFG.model_dir + f"{CFG.model_name}_fold{fold}_{CFG.inf_weight}.pth",
        map_location=torch.device("cpu"),
    )
    mask_pred = check_point["preds"]
    best_dice, best_th = calc_fbeta(valid_mask_gt, mask_pred, Logger)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].imshow(valid_mask_gt)
    axes[1].imshow(mask_pred)
    axes[2].imshow((mask_pred >= best_th).astype(int))


if __name__ == "__main__":
    main()
