import os, sys, argparse, logging, random, numpy as np, matplotlib.pyplot as plt
import time
import SimpleITK as sitk
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from unet2d import UNet16, UNet32, UNet64 
from dataloaders import ACDCTrainDataset
from utils import create_if_not, set_random, test_single_case_dc, test_single_volume_slicebyslice
from loss import DiceCeLoss


# -------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='../log/ACDC_final')
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='../outputs_ACDC')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--rot', type=int, default=1)
    return parser.parse_args()


# -------------------------------------------------------
def plot_metrics(history, save_path, metric_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history[f"train_{metric_name}"], label=f"Train {metric_name}")
    plt.plot(history[f"val_{metric_name}"], label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric_name}_curve.png"))
    plt.close()


# -------------------------------------------------------
def main(args):
    print('****** make logger ******')
    snapshot_path = args.log_dir
    create_if_not(snapshot_path)
    save_model_path = os.path.join(snapshot_path, 'model')
    create_if_not(save_model_path)

    logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ---- Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = UNet16(in_chns=1, class_num=args.num_class).to(device)
    #model = UNet32(in_chns=1, class_num=args.num_class).to(device)
    model = UNet64(in_chns=1, class_num=args.num_class).to(device)
    loss_fn = DiceCeLoss(args.num_class)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',        
        factor=0.2,        
        patience=5,        
        threshold=1e-4,    
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-6,
    )

    train_dataset = ACDCTrainDataset(
        args.data_dir,
        flip=bool(args.flip),
        rot=bool(args.rot),
        brightness_delta=0.05,      # 0.05 intensity
        contrast_range=(0.9, 1.1),  # 10% contrast
        gamma_range=(0.9, 1.1),     # gamma variation
        noise_std=0.01              # Gaussian noise
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation setup
    val_path = os.path.join(args.data_dir, 'volume')
    with open(os.path.join(args.data_dir, 'val.txt'), 'r') as f:
        val_list = [os.path.join(val_path, x.strip() + '.nii.gz') for x in f.readlines()]
    print(f"Validation set has {len(val_list)} volumes.")

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    best_dice = 0
    best_epoch = 0

    # early stopping setup
    patience = 15
    counter = 0
    early_stop = False

    print('****** start training ******')
    start_time = time.time()
    for epoch in range(args.max_epoch):
        # -------------------- TRAIN --------------------
        model.train()
        train_losses, train_dices = [], []

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = loss_fn(outputs, label.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labs = label.cpu().numpy()
                batch_dice = [np.mean(test_single_case_dc(p, l, args.num_class)) for p, l in zip(preds, labs)]
                train_dices.extend(batch_dice)

        avg_train_loss = np.mean(train_losses)
        avg_train_dice = np.mean(train_dices)
        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)

        # -------------------- VALIDATION --------------------
        model.eval()
        val_dices, val_losses = [], []

        with torch.no_grad():
            for val_img in val_list:
                img_path = val_img
                seg_path = img_path.replace('.nii.gz', '_gt.nii.gz')
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
                seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(np.int8)
                img = (img - np.mean(img)) / np.std(img)
                pred = test_single_volume_slicebyslice(img, model)
                dice_arr = test_single_case_dc(pred, seg, args.num_class)
                val_dices.append(np.mean(dice_arr))

                # validation loss slice-wise
                input_t = torch.tensor(img).unsqueeze(1).to(device, dtype=torch.float32)
                label_t = torch.tensor(seg).to(device, dtype=torch.long)
                output_t = model(input_t)
                val_loss = loss_fn(output_t, label_t.unsqueeze(1))
                val_losses.append(val_loss.item())

        avg_val_dice = np.mean(val_dices)
        avg_val_loss = np.mean(val_losses)
        history['val_dice'].append(avg_val_dice)
        history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_dice)

        # -------------------- LOGGING --------------------
        logging.info(f"Epoch [{epoch+1}/{args.max_epoch}] "
                     f"Train loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | "
                     f"Val loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        writer.add_scalar('loss/train_loss', avg_train_loss, epoch)
        writer.add_scalar('loss/val_loss', avg_val_loss, epoch)
        writer.add_scalar('dice/train_dice', avg_train_dice, epoch)
        writer.add_scalar('dice/val_dice', avg_val_dice, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # save best
        if avg_val_dice >= best_dice:
            best_dice = avg_val_dice
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best.pth'))
        else:
            counter += 1
            logging.info(f"No improvement in val Dice for {counter} epochs.")
            if counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Best val Dice: {best_dice:.4f} at epoch {best_epoch+1}")
                early_stop = True
                break


        torch.save(model.state_dict(), os.path.join(save_model_path, f'epoch_{epoch}.pth'))
        


        # update plots
        plot_metrics(history, snapshot_path, 'loss')
        plot_metrics(history, snapshot_path, 'dice')
        
    if early_stop:
         print(f"Early stopping at epoch {epoch+1}. Best val Dice: {best_dice:.4f} (epoch {best_epoch+1})")
    writer.close()
    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"****** Training finished in {hours}h {minutes}m {seconds}s ******")
    logging.info(f"Training time: {hours}h {minutes}m {seconds}s")

    print(f"****** Training finished. Best epoch: {best_epoch}, Best mean val Dice: {best_dice:.4f} ******")


# -------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    create_if_not(args.log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"****** fix random seed {args.seed} ******")
    set_random(args.seed)
    main(args)
