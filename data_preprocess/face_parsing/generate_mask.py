import torch
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import cv2
from mask_model import BiSeNet
from read_dataset import DatasetMask
import joblib
import tqdm
import argparse


def vis_parsing_maps(im, parsing_anno, stride, save_im, save_path):
    # Colors for all 19 parts
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    tt_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        tt_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        tt_anno_color[index[0], index[1], :] = part_colors[pi]
        cv2.imwrite(save_path[:-4] + 'ano_{}.jpg'.format(pi), tt_anno_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # print('anno',vis_parsing_anno_color)
        cv2.imwrite(save_path[:-4] + 'ano.jpg', vis_parsing_anno_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im


def generate_main(model_path, root_dir, all_live_data, batchsize=32):
    gpus = [str(i) for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = DatasetMask(all_live_data, root_dir, transform=to_tensor)

    kwargs = {'batch_size': batchsize * torch.cuda.device_count(), 'num_workers': 1, 'shuffle': False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # net = torch.nn.DataParallel(net).cuda()
    # model_path = "79999_iter.pth"          # load from https://github.com/zllrunning/face-parsing.PyTorch
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    print("---load model successfully---")

    with torch.no_grad():
        with tqdm.tqdm(total=len(train_loader), desc='Iterate over data') as pbar:
            for img_paths, data in train_loader:
                data = data.cuda()
                out = net(data)[0]
                masks = out.argmax(1).cpu().numpy()
                # print(masks.shape)

                for i, mask in enumerate(masks):
                    img_path = img_paths[i]
                    img_name = img_path.split("/")[-1].split(".")[0]
                    origin_image = cv2.imread(os.path.join(root_dir, "cvpr2024/data", img_path))
                    img_shape = origin_image.shape
                    protocal = img_path.split("/")[0]
                    mask_store_path = os.path.join(root_dir, "cvpr2024/data", protocal, "train_live_mask", img_name + '.pkl')

                    mask = mask.astype(np.uint8)
                    mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
                    joblib.dump(mask, mask_store_path)

                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate mask")
    parser.add_argument('--root_dir', type=str,
                        default='xxx')
    parser.add_argument('--model_path', type=str,
                        default='79999_iter.pth')
    args = parser.parse_args()

    root_dir = args.root_dir
    model_path = args.model_path

    protocols = ['p1', 'p2.1', 'p2.2']
    all_live_data = []
    for protocol in protocols:
        lines = open(os.path.join(root_dir, f'cvpr2024/data/{protocol}/train_label.txt'), "r").readlines()
        for line in lines:
            path, label = line.strip().split()
            if int(label) == 0:
                all_live_data.append(path)

        mask_save_path = os.path.join(root_dir, "cvpr2024/data", protocol, "train_live_mask")
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)

    generate_main(model_path, root_dir, all_live_data, batchsize=32)



