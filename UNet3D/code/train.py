import os, sys, time, argparse, logging, numpy as np, torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from functools import partial
from tensorboardX import SummaryWriter

from utils import set_random, create_if_not, LOG_EVERY_EPOCH, IGNORE_INDEX
from dataloaders import ACDCVolume3D, collate_pad_depth
from unet3d import UNet3D
from utils import apply_pad_inv, IGNORE_INDEX
from loss import DiceCELoss3D, dice_per_class_np


@torch.no_grad()
def validate(model, loader, device, C):
    model.eval()
    criterion = DiceCELoss3D(n_class=C, ignore_index=IGNORE_INDEX)  
    losses, dices = [], []

    for (img_t, seg_t, pads, vids) in loader:
        img_t = img_t.to(device, non_blocking=True)
        seg_t = seg_t.to(device, non_blocking=True)

        valid_range = (seg_t >= 0) & (seg_t < C)          
        ignore_mask = seg_t.eq(IGNORE_INDEX)
        ok = valid_range | ignore_mask
        if not ok.all():
            seg_t = torch.where(ok, seg_t, torch.full_like(seg_t, IGNORE_INDEX))

        with autocast('cuda'):
            logits = model(img_t)
            loss = criterion(logits, seg_t)
        losses.append(loss.item())

        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  
        gt   = seg_t.squeeze(0).cpu().numpy()

        pred_hwz = np.transpose(pred, (1,2,0))
        gt_hwz   = np.transpose(gt,   (1,2,0))
        pad = pads[0]
        pred_hwz = apply_pad_inv(pred_hwz, pad)
        gt_hwz   = apply_pad_inv(gt_hwz,   pad)
        pred = np.transpose(pred_hwz, (2,0,1))
        gt   = np.transpose(gt_hwz,   (2,0,1))

        dices.append(dice_per_class_np(pred, gt, C).mean())

    return float(np.mean(losses)), float(np.mean(dices))


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_dir', type=str, default='../outputs_ACDC')
    ap.add_argument('--log_dir', type=str, default='../log/ACDC_3D')
    ap.add_argument('--num_class', type=int, default=4)
    ap.add_argument('--base_ch', type=int, default=26)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max_epoch', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=4)    
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--accum_steps', type=int, default=1)   # grad acc possible
    return ap.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    create_if_not(args.log_dir); create_if_not(os.path.join(args.log_dir, 'model'))

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    set_random(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vol_dir = os.path.join(args.data_dir, 'volume')
    all_ids = [f[:-7] for f in os.listdir(vol_dir)
               if f.endswith('.nii.gz') and not f.endswith('_gt.nii.gz')]
    with open(os.path.join(args.data_dir, 'val.txt'), 'r') as f:
        val_ids = [x.strip() for x in f.readlines()]
    train_ids = [i for i in all_ids if i not in val_ids]

    print(f"Train vols: {len(train_ids)} | Val vols: {len(val_ids)}")

    train_ds = ACDCVolume3D(args.data_dir, train_ids, augment=True)
    val_ds   = ACDCVolume3D(args.data_dir, val_ids,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=partial(collate_pad_depth, ignore_index=IGNORE_INDEX)
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=partial(collate_pad_depth, ignore_index=IGNORE_INDEX)
    )

    model = UNet3D(in_ch=1, n_class=args.num_class, base=args.base_ch).to(device)
    criterion = DiceCELoss3D(n_class=args.num_class, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6, verbose=True
    )
    scaler = GradScaler('cuda')
    writer = SummaryWriter(os.path.join(args.log_dir, 'tb'))

    best_dice, best_epoch, no_improve = 0.0, -1, 0
    print("****** start training ******")
    t0 = time.time()

    for epoch in range(args.max_epoch):
        model.train()
        running = []
        step_in_accum = 0
        optimizer.zero_grad(set_to_none=True)

        for (img_t, seg_t, pads, vids) in train_loader:
            assert (seg_t.eq(IGNORE_INDEX) | ((seg_t >= 0) & (seg_t < args.num_class))).all(), \
                f"Label out of range; expected 0..{args.num_class-1} or {IGNORE_INDEX}"

            img_t = img_t.to(device, non_blocking=True)  
            seg_t = seg_t.to(device, non_blocking=True)  

            valid_range = (seg_t >= 0) & (seg_t < args.num_class)
            ignore_mask = seg_t.eq(IGNORE_INDEX)
            ok = valid_range | ignore_mask
            if not ok.all():
                # replace bad vals 
                seg_t = torch.where(ok, seg_t, torch.full_like(seg_t, IGNORE_INDEX))

            with autocast('cuda'):
                logits = model(img_t)
                loss = criterion(logits, seg_t) / args.accum_steps

            scaler.scale(loss).backward()
            step_in_accum += 1
            if step_in_accum % args.accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0

            running.append(loss.item() * args.accum_steps)

        # validation
        val_loss, val_dice = validate(model, val_loader, device, args.num_class)
        scheduler.step(val_dice)

        train_loss = float(np.mean(running))
        lr_now = optimizer.param_groups[0]['lr']

        if (epoch + 1) % LOG_EVERY_EPOCH == 0:
            logging.info(f"Epoch [{epoch+1}/{args.max_epoch}] "
                         f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValDice {val_dice:.4f} | LR {lr_now:.2e}")
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val',   val_loss,   epoch)
            writer.add_scalar('dice/val',   val_dice,   epoch)
            writer.add_scalar('lr',         lr_now,     epoch)

        # early stop and save best
        if val_dice >= best_dice:
            best_dice, best_epoch, no_improve = val_dice, epoch, 0
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model', 'best.pth'))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1}. Best val Dice {best_dice:.4f} @ {best_epoch+1}")
                break

        # epoch cpt
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'model', f'epoch_{epoch:03d}.pth'))

    dt = time.time() - t0
    h, m, s = int(dt//3600), int((dt%3600)//60), int(dt%60)
    logging.info(f"Finished in {h}h {m}m {s}s. Best Dice {best_dice:.4f} @ epoch {best_epoch+1}")
    writer.close()


if __name__ == "__main__":
    args = get_args()
    create_if_not(args.log_dir)
    main(args)
