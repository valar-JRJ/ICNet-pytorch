import yaml
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data

from dataset.sunrgbd_loader import SUNRGBDLoader
from models.icnet import ICNet
from utils.metric import runningScore
from utils.utils import convert_state_dict

torch.backends.cudnn.benchmark = True


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_path = cfg["train"]["data_path"]

    loader = SUNRGBDLoader(
        data_path,
        split='val',
        is_transform=True,
        img_norm=True,
        img_size=(cfg["train"]["img_rows"], cfg["train"]["img_cols"]),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=cfg["train"]["valid_batch_size"], num_workers=8)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = ICNet(nclass=n_classes).to(device)
    # state = convert_state_dict(torch.load(args.model_path)["model_state"])
    state = torch.load(args.model_path)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = images.to(device)

        if args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs[0].data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = torch.argmax(outputs, 1)
        else:
            outputs = model(images)
            pred = outputs[0].data.max(1)[1].cpu().numpy()

        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/icnet-sunrgbd.yaml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="icnet_best_model.pth",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=False)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
