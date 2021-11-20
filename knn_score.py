#!/usr/bin/env python3
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
# from scipy.misc import imread
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d

import pickle
from scipy.stats import multivariate_normal
from sklearn import mixture
import torch.nn.functional as F


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=1,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--pca_path', type=str, default=None)
# "/mnt/blob/code/image-judge/gaussian/pca_stat/pca_all_95.pkl"
parser.add_argument('--act_path', type=str, default="/mnt/blob/code/image-judge/gaussian/statistic/cat_act_sample_200000.pkl")
parser.add_argument('--output_file', type=str, default="/mnt/blob/datasets/generation_results/score_results/try_out.txt")
parser.add_argument('--K', type=int, default=1)

def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB'), dtype=np.uint8)[..., :3]

def get_activations(files, model, batch_size, dims, cuda, verbose, pca_path, act_path, output_file, K):
    
    model.eval()

    batch_size = 50

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    act = pickle.load(open(act_path, "rb"))
    act = torch.from_numpy(act).cuda()
    act = act.type(torch.cuda.FloatTensor)
    print(act.size())
    file_path = output_file

    if pca_path != None:
        pca = pickle.load(open(pca_path, "rb"))

    score_list = []
    
    with open(file_path, 'wt') as f:
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = start + batch_size 
            images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])
        
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            images /= 255
        
            batch = torch.from_numpy(images).type(torch.FloatTensor)
            if cuda:
                batch = batch.cuda()
            with torch.no_grad():
                pred = model(batch)[0]
        
            if pca_path != None:
                pred = pca.transform(pred[:,:,0,0].cpu().numpy()) 
                pred = torch.from_numpy(pred).type_as(act)
    
            for image_i in range(0, batch_size):
                if pca_path == None:
                    this_pred = pred[image_i,:,0,0].unsqueeze(0).repeat(act.size()[0],1)
                else:
                    this_pred = pred[image_i,:].unsqueeze(0).repeat(act.size()[0],1)
                
                dis = F.pairwise_distance(this_pred, act, p=1)
                dis_list = dis.tolist()
                sort_list = sorted(dis_list)
                k_num = K
                score = 0
                for k in range(0,k_num):
                    score = score + 1/sort_list[k]
                image_file = str(files[start+image_i]).split('/')[-1]
                f.write("score of "+image_file+" is:\n")
                f.write(str(score))
                f.write("\n")

    return pred_arr


def calculate_activation_statistics(files, model, batch_size, dims, cuda, pca_path, act_path, output_file, K):
    verbose = False
    act = get_activations(files, model, batch_size, dims, cuda, verbose, pca_path, act_path, output_file, K)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, pca_path, act_path, output_file, K):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()

    else:
        path = pathlib.Path(path)

        # image_file = open(path)
        # files = image_file.readlines()

        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda, pca_path, act_path, output_file, K)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims, pca_path, act_path, output_file, K):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda, pca_path, act_path, output_file, K)

    return 777


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value = calculate_fid_given_paths(args.path, args.batch_size, args.gpu != '', args.dims, args.pca_path, args.act_path, args.output_file, args.K)
    print('FID: ', fid_value)
