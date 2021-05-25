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
from utils import ICNetLoss, IterationPolyLR, runningScore, averageMeter, SetupLogger, ConstantLR
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
                                      img_size=(cfg['train']['img_rows'], cfg['train']['img_cols']),
                                      img_norm=True
                                      )
        val_dataset = SUNRGBDLoader(root=cfg["train"]["data_path"],
                                    split="val",
                                    is_transform=True,
                                    img_size=(cfg['train']['img_rows'], cfg['train']['img_cols']),
                                    img_norm=True
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
        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=[20, 40, 60, 80, 100, 120, 140],
                                        gamma=0.5)
        # self.lr_scheduler = ConstantLR(self.optimizer)

        # dataparallel
        if self.dataparallel:
            self.model = nn.DataParallel(self.model)

        # evaluation metrics
        self.metric = runningScore(train_dataset.n_classes)

        self.current_mIoU = 0.0
        self.best_mIoU = 0.0
        
        self.epochs = cfg["train"]["epochs"]
        self.current_epoch = 0
        self.current_iteration = 0

        if cfg["train"]["resume"] is not None:
            if os.path.isfile(cfg["train"]["resume"]):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["train"]["resume"])
                )
                checkpoint = torch.load(cfg["train"]["resume"])
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.current_epoch = checkpoint["epoch"]
                logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        cfg["train"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                logger.info("No checkpoint found at '{}'".format(cfg["train"]["resume"]))
        
    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        
        time_meter = averageMeter()
        train_loss_meter = averageMeter()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        
        self.model.train()
        
        # for _ in range(self.epochs):
        while self.current_epoch <= self.epochs:
            self.current_epoch += 1
            self.lr_scheduler.step()
            # for i, (images, targets, _) in enumerate(self.train_dataloader):
            for i, (images, targets) in enumerate(self.train_dataloader):  
                self.current_iteration += 1
                start_time = time.time()

                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                pred = outputs[0].data.max(1)[1].cpu().numpy()
                gt = targets.data.cpu().numpy()

                self.metric.update(gt, pred)
                train_loss_meter.update(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                time_meter.update(time.time() - start_time)
                
                if self.current_iteration % log_per_iters == 0:
                    eta_seconds = time_meter.avg * (max_iters - self.current_iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logger.info(
                        "Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            self.current_epoch, self.epochs, 
                            self.current_iteration, max_iters, 
                            self.optimizer.param_groups[0]['lr'], 
                            loss.item(),
                            str(datetime.timedelta(seconds=int(time_meter.val))),
                            eta_string))
                    time_meter.reset()

            writer.add_scalar("loss/train_loss", train_loss_meter.avg, self.current_epoch)
            score, class_iou = self.metric.get_scores()
            for k, v in score.items():
                print(k, v)
                logger.info("{}: {}".format(k, v))
                writer.add_scalar("train_metrics/{}".format(k), v, self.current_epoch)

            for k, v in class_iou.items():
                logger.info("{}: {}".format(k, v))

            self.metric.reset()
            train_loss_meter.reset()

            if self.current_iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

    def validation(self):
        is_best = False
        if self.dataparallel:
            model = self.model.module
        else:
            model = self.model
        model.eval()
        val_loss_meter = averageMeter()
        # for i, (image, targets, filename) in enumerate(self.val_dataloader):
        for i, (image, targets) in enumerate(self.val_dataloader):
            image = image.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                outputs = model(image)
                loss = self.criterion(outputs, targets)
                pred = outputs[0].data.max(1)[1].cpu().numpy()
                gt = targets.data.cpu().numpy()

            self.metric.update(gt, pred)
            val_loss_meter.update(loss.item())

        logger.info("epoch %d Loss: %.4f" % (self.current_epoch, val_loss_meter.avg))
        writer.add_scalar("loss/val_loss", val_loss_meter.avg, self.current_epoch)
        score, class_iou = self.metric.get_scores()
        for k, v in score.items():
            logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, self.current_epoch)

        for k, v in class_iou.items():
            logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/cls_{}".format(k), v, self.current_epoch)
        self.current_mIoU = score["Mean IoU : \t"]
        logger.info("Validation: Average loss: {:.3f}, mIoU: {:.3f}, mean pixAcc: {:.3f}"
                    .format(val_loss_meter.avg,  self.current_mIoU, score["Mean Acc : \t"]))
        self.metric.reset()
        val_loss_meter.reset()

        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save Checkpoint"""
        # directory = os.path.expanduser(cfg["train"]["ckpt_dir"])
        directory = logdir
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = '{}_{}_{}_{:.3f}.pth'.format(cfg["model"]["name"],
                                                cfg["model"]["backbone"],
                                                self.current_epoch,
                                                self.current_mIoU)
        filename = os.path.join(directory, filename)
        if self.dataparallel:
            model = self.model.module

            state = {
                "epoch": self.current_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.lr_scheduler.state_dict(),
                "best_iou": self.best_mIoU,
            }
            best_filename = os.path.join(directory, filename)
            torch.save(state, best_filename)
        

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
