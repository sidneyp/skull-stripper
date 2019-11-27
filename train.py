import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import random
import json
import time
from logger import Logger
from loss import DiceLoss
from utils import log_images
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
import argparse
from dataset import MouseMRIDS, load_dataset

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main(args):
    def log_loss_summary(logger, loss, step, prefix=""):
        logger.scalar_summary(prefix + "loss", np.mean(loss), step)

    def accuracy_f(predict, labels):
        return (predict.round() == labels).float().sum() / predict.shape[0]

    def log_accuracy_summary(logger, loss, step, prefix=""):
        logger.scalar_summary(prefix + "accuracy", np.mean(loss), step)

    def log_prediction_summary(logger, loss, step, prefix=""):
        logger.scalar_summary(prefix + "prediction", np.mean(loss), step)

    def log_recall_summary(logger, loss, step, prefix=""):
        logger.scalar_summary(prefix + "recall", np.mean(loss), step)

    def log_f1score_summary(logger, loss, step, prefix=""):
        logger.scalar_summary(prefix + "f1score", np.mean(loss), step)

    # Load the data
    src_train, msk_train, src_val, msk_val = load_dataset(
            args.src_path, args.mask_path, args.validation_portion)

    trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])
    train = MouseMRIDS(src_train, msk_train,transform=trans)
    val = MouseMRIDS(src_val, msk_val,transform=trans, augmentation=False)
    val_augmentation = MouseMRIDS(src_val, msk_val,transform=trans,augmentation=True)

    bt_sz = args.batch_size

    training = DataLoader(train, batch_size=bt_sz, shuffle=False)
    validating = DataLoader(val, batch_size=bt_sz, shuffle=False)
    validating_aug = DataLoader(val_augmentation, batch_size=bt_sz,
                                shuffle=False)

    # Prepare and instantiate the model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32,
        pretrained=args.is_pretrained)
    model.cuda()

    # Instantiate the loss class
    if args.loss == "dice":
        loss_f = DiceLoss()
    elif args.loss == "bce":
        loss_f = torch.nn.BCELoss()
    else:
        raise Exception('loss argument should be dice or bce')

    lr = args.lr
    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "log_"+timestr)

    logger = Logger(log_dir)

    with open(os.path.join(log_dir,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if not os.path.exists(args.weights):
        os.makedirs(args.weights)

    loss_train = []
    accuracy_train = []
    prediction_train = []
    recall_train = []
    f1score_train = []
    best_val_loss = np.inf

    # Train the model
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        for inputs, labels in training:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            predict = model(inputs)
            predicti = predict.permute(0, 2, 3, 1)
            labeli = labels.permute(0, 2, 3, 1)
            batch_sz, width, height, ch = predicti.shape
            predicti_rsz = predicti.resize(batch_sz*width*height, 1)
            labeli_rsz = labeli.resize(batch_sz*width*height, 1)

            loss = loss_f(predicti_rsz, labeli_rsz)
            loss_train.append(loss.item())

            accuracy = accuracy_f(predicti_rsz, labeli_rsz)
            accuracy_train.append(accuracy.item())

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predicti_rsz_cpu = (predicti_rsz.cpu().numpy() > 0.5).astype(int)
                labeli_rsz_cpu = labeli_rsz.cpu().numpy().astype(int)

                prediction_train.append(precision_score(predicti_rsz_cpu, labeli_rsz_cpu))
                recall_train.append(recall_score(predicti_rsz_cpu, labeli_rsz_cpu))
                f1score_train.append(f1_score(predicti_rsz_cpu, labeli_rsz_cpu))

        # ===================log========================
        scheduler.step()
        loss_cpu = loss.data.cpu()
        l = loss_cpu.numpy()
        l.max()
        log_loss_summary(logger, loss_train, epoch)
        log_accuracy_summary(logger, accuracy_train, epoch)
        log_prediction_summary(logger, prediction_train, epoch)
        log_recall_summary(logger, recall_train, epoch)
        log_f1score_summary(logger, f1score_train, epoch)

        loss_train = []
        accuracy_train = []
        prediction_train = []
        recall_train = []
        f1score_train = []
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, l.max()))

        if epoch % 30 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir,'skull-stripper-chkpoint-{0:05d}.pth'.format(epoch)))

            list_val_loss = []
            for inputs, labels in validating:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    predict = model(inputs)

                    predicti = predict.permute(0, 2, 3, 1)
                    labeli = labels.permute(0, 2, 3, 1)
                    batch_sz, width, height, ch = predicti.shape
                    predicti_rsz = predicti.resize(batch_sz*width*height, 1)
                    labeli_rsz = labeli.resize(batch_sz*width*height, 1)
                    predicti_rsz_cpu = (predicti_rsz.cpu().numpy() > 0.5).astype(int)
                    labeli_rsz_cpu = labeli_rsz.cpu().numpy().astype(int)

                    loss_valid = loss_f(predicti_rsz, labeli_rsz)
                    log_loss_summary(logger, loss_valid.item(), epoch, prefix="val_")

                    accuracy_valid = accuracy_f(predicti_rsz, labeli_rsz)
                    log_accuracy_summary(logger, accuracy_valid.item(), epoch, prefix="val_")

                    log_prediction_summary(logger, precision_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_")
                    log_recall_summary(logger, recall_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_")
                    log_f1score_summary(logger, f1_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_")

                    tag = "image/{}".format(epoch)
                    list_val_loss.append(loss_valid.item())
                    logger.image_list_summary(
                        tag,
                        log_images(inputs, labels, predict),
                        epoch,
                        )
            current_val_loss = np.mean(list_val_loss)
            if best_val_loss > current_val_loss:
                torch.save(model.state_dict(), os.path.join(args.weights,'skull-stripper.pth'))

            for inputs, labels in validating_aug:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    predict = model(inputs)

                    predicti = predict.permute(0, 2, 3, 1)
                    labeli = labels.permute(0, 2, 3, 1)
                    batch_sz, width, height, ch = predicti.shape
                    predicti_rsz = predicti.resize(batch_sz*width*height, 1)
                    labeli_rsz = labeli.resize(batch_sz*width*height, 1)
                    predicti_rsz_cpu = (predicti_rsz.cpu().numpy() > 0.5).astype(int)
                    labeli_rsz_cpu = labeli_rsz.cpu().numpy().astype(int)

                    loss_valid = loss_f(predicti_rsz, labeli_rsz)
                    log_loss_summary(logger, loss_valid.item(), epoch, prefix="val_aug_")

                    accuracy_valid = accuracy_f(predicti_rsz, labeli_rsz)
                    log_accuracy_summary(logger, accuracy_valid.item(), epoch, prefix="val_aug_")

                    log_prediction_summary(logger, precision_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_aug_")
                    log_recall_summary(logger, recall_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_aug_")
                    log_f1score_summary(logger, f1_score(predicti_rsz_cpu, labeli_rsz_cpu), epoch, prefix="val_aug_")

                    tag = "image_aug/{}".format(epoch)
                    list_val_loss.append(loss_valid)
                    logger.image_list_summary(
                        tag,
                        log_images(inputs, labels, predict),
                        epoch,
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="input batch size for training (default: 25)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr-step",
        type=int,
        default=200,
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dice",
        help="loss function type list: [dice, bce] (default: dice)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--src-path", type=str, default="./source_images", help="folder of source images as .mat"
    )
    parser.add_argument(
        "--mask-path", type=str, default="./masks", help="folder of masks as .nii"
    )
    parser.add_argument(
        "--validation-portion",
        type=float,
        default=0.05,
        help="data portion as validation data set (default: 0.05)",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="is_pretrained",
        action="store_false",
        help="train without pretrained model",
    )

    args = parser.parse_args()
    main(args)
