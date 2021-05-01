import os
import time
import datetime
import yaml
import random
import shutil
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter

# from dataset import CityscapesDataset
from dataset.sunrgbd_loader import SUNRGBDLoader
from models import ICNet
from utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from torch.optim.lr_scheduler import MultiStepLR


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataparallel = torch.cuda.device_count() > 1
        
        # dataset and dataloader
        # train_dataset = CityscapesDataset(root = cfg["train"]["cityscapes_root"], 
        #                                   split='train', 
        #                                   base_size=cfg["model"]["base_size"], 
        #                                   crop_size=cfg["model"]["crop_size"])
        # val_dataset = CityscapesDataset(root = cfg["train"]["cityscapes_root"], 
        #                                 split='val',
        #                                 base_size=cfg["model"]["base_size"], 
        #                                 crop_size=cfg["model"]["crop_size"])
        train_dataset = SUNRGBDLoader(root=cfg["train"]["data_path"],
                                      split="training",
                                      is_transform=True,
                                      img_size=(480, 480)
                                      )
        val_dataset = SUNRGBDLoader(root=cfg["train"]["data_path"],
                                    split="val",
                                    is_transform=True,
                                    img_size=(480, 480)
                                    )
        self.train_dataloader = data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg["train"]["train_batch_size"],
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True,
                                                drop_last=False)
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=False)
        
        self.iters_per_epoch = len(self.train_dataloader)
        self.max_iters = cfg["train"]["epochs"] * self.iters_per_epoch

        # create network
        self.model = ICNet(nclass=train_dataset.n_classes, backbone='resnet50').to(self.device)
        
        # create criterion
        # self.criterion = ICNetLoss(ignore_index=train_dataset.IGNORE_INDEX).to(self.device)
        self.criterion = ICNetLoss(ignore_index=-1).to(self.device)
        
        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': cfg["optimizer"]["init_lr"]})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': cfg["optimizer"]["init_lr"] * 10})
        self.optimizer = torch.optim.SGD(params=params_list,
                                         lr=cfg["optimizer"]["init_lr"],
                                         momentum=cfg["optimizer"]["momentum"],
                                         weight_decay=cfg["optimizer"]["weight_decay"])
        # self.optimizer = torch.optim.SGD(params = self.model.parameters(),
        #                                  lr = cfg["optimizer"]["init_lr"],
        #                                  momentum=cfg["optimizer"]["momentum"],
        #                                  weight_decay=cfg["optimizer"]["weight_decay"])
        
        # lr scheduler
        # self.lr_scheduler = IterationPolyLR(self.optimizer, max_iters=self.max_iters, power=0.9)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[26000, 30000], gamma=0.1)

        # dataparallel
        if(self.dataparallel):
            self.model = nn.DataParallel(self.model)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.n_classes)

        self.current_mIoU = 0.0
        self.best_mIoU = 0.0
        
        self.epochs = cfg["train"]["epochs"]
        self.current_epoch = 0
        self.current_iteration = 0

        if cfg["train"]["resume"] is not None:
            if os.path.isfile(cfg["train"]["resume"]):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
                checkpoint = torch.load(cfg["training"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.current_epoch = checkpoint["epoch"]
                logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
        
    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        
        self.model.train()
        
        for _ in range(self.epochs): 
            self.current_epoch += 1
            lsit_pixAcc = []
            list_mIoU = []
            list_loss = []
            self.metric.reset()
            # for i, (images, targets, _) in enumerate(self.train_dataloader):
            for i, (images, targets) in enumerate(self.train_dataloader):  
                self.current_iteration += 1
                
                self.lr_scheduler.step()

                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.metric.update(outputs[0], targets)
                pixAcc, mIoU = self.metric.get()
                lsit_pixAcc.append(pixAcc)
                list_mIoU.append(mIoU)
                list_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (max_iters - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if self.current_iteration % log_per_iters == 0:
                    logger.info(
                        "Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            self.current_epoch, self.epochs, 
                            self.current_iteration, max_iters, 
                            self.optimizer.param_groups[0]['lr'], 
                            loss.item(), 
                            mIoU,
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), 
                            eta_string))

            average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
            average_mIoU = sum(list_mIoU)/len(list_mIoU)
            average_loss = sum(list_loss)/len(list_loss)
            logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
            writer.add_scalar("loss/train_loss", average_loss, self.current_epoch)
            writer.add_scalar("miou/train_miou", average_mIoU, self.current_epoch)
            writer.add_scalar("pixacc/train_pixacc", average_pixAcc, self.current_epoch)

            if self.current_iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
            total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.dataparallel:
            model = self.model.module
        else:
            model = self.model
        model.eval()
        lsit_pixAcc = []
        list_mIoU = []
        list_loss = []
        # for i, (image, targets, filename) in enumerate(self.val_dataloader):
        for i, (image, targets) in enumerate(self.val_dataloader):
            image = image.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                outputs = model(image)
                loss = self.criterion(outputs, targets)
            self.metric.update(outputs[0], targets)
            pixAcc, mIoU = self.metric.get()
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            list_loss.append(loss.item())

        average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU)/len(list_mIoU)
        average_loss = sum(list_loss)/len(list_loss)
        self.current_mIoU = average_mIoU
        logger.info("Validation: Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(average_loss,  average_mIoU, average_pixAcc))
        writer.add_scalar("loss/val_loss", average_loss, self.current_epoch)
        writer.add_scalar("miou/val_miou", average_mIoU, self.current_epoch)
        writer.add_scalar("pixacc/val_pixacc", average_pixAcc, self.current_epoch)

        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            save_checkpoint(self.model, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.dataparallel)


def save_checkpoint(model, cfg, epoch=0, is_best=False, mIoU=0.0, dataparallel=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg["train"]["ckpt_dir"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{:.3f}.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch, mIoU)
    filename = os.path.join(directory, filename)
    if dataparallel:
        model = model.module
    if is_best:
        best_filename = '{}_{}_{}_{:.3f}_best_model.pth'.format(cfg["model"]["name"],
                                                                cfg["model"]["backbone"],
                                                                epoch,
                                                                mIoU)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)
        

if __name__ == '__main__':
    # Set config file
    config_path = "./configs/icnet-sunrgbd.yaml"
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
        #print(cfg)
        #print(cfg["model"]["backbone"])
        #print(cfg["train"]["specific_gpu_num"])
    
    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    # print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", 'icnet-sunrgbd', str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    # Set logger
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=logdir,
                         distributed_rank=0,
                         filename='{}_{}_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    # logger.info("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    # logger.info("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    logger.info(cfg)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(config_path, logdir)

    # Start train
    trainer = Trainer(cfg)
    trainer.train()
