import os
import torch
import glob
import argparse
import imageio
import numpy as np
from PIL import Image

from models.icnet import ICNet
from dataset.sunrgbd_loader import SUNRGBDLoader
from utils.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args, img_path, device, loader, model):
    # Setup image
    print("Read Input Image from : {}".format(args.folder_path))
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)

    resized_img = np.array(Image.fromarray(img).resize((loader.img_size[0], loader.img_size[1]), Image.BICUBIC))

    orig_size = img.shape[:-1]

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)

    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + "_drf.png"
        Image.fromarray(decoded_crf).save(dcrf_path)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs[0].data.max(1)[1].cpu().numpy(), axis=0)
    pred = pred.astype(np.float32)
    # float32 with F mode, resize back to orig_size
    # pred = misc.imresize(pred, orig_size, "nearest", mode="F")
    pred = np.array(Image.fromarray(pred, mode='F').resize((orig_size[1], orig_size[0]), Image.NEAREST))

    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    filename = img_path.split('/')[-1].split('.')[0]
    imageio.imwrite('output/{}output.png'.format(filename), decoded)
    # Image.fromarray(decoded).save('output/{}output.png'.format(filename))
    # misc.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/icnet-sunrgbd/46739/icnet_resnet50_150_0.270_best_model.pth",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--folder_path", nargs="?", type=str, default='../data/samples', help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default='output', help="Path of the output segmap"
    )
    opt = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_loader = SUNRGBDLoader(root=None,
                               is_transform=True,
                               img_norm=opt.img_norm,
                               img_size=(520, 520),
                               test_mode=True)
    n_classes = img_loader.n_classes

    # Setup Model
    model = ICNet(nclass=n_classes, backbone='resnet50')
    # state = convert_state_dict(torch.load(opt.model_path)["model_state"])
    state = torch.load(opt.model_path)
    model.load_state_dict(state)
    model.eval()
    model.to(dev)

    label_path = os.path.join(opt.folder_path, 'labels')
    files = sorted(glob.glob("%s/*.*" % opt.folder_path))
    for image_path in files:
        test(opt, image_path, dev, img_loader, model)
    files = sorted(glob.glob("%s/*.*" % label_path))
    for path in files:
        lbl = Image.open(path)
        lbl = np.array(lbl, dtype=np.uint8)
        gt = img_loader.decode_segmap(lbl)
        filename = path.split('/')[-1].split('.')[0]
        imageio.imwrite('output/{}gt.png'.format(filename), gt)