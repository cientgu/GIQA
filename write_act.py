#!/usr/bin/env python3
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.decomposition import PCA
import pickle

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
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--pca_rate', default=0.95, type=float)
parser.add_argument('--pca_path', default=None, type=str)
parser.add_argument('--act_path', default=None, type=str)


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    model.eval()

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

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr



def calculate_activation_statistics(files, model, batch_size, dims, cuda, pca_path, pca_rate, act_path):
    act = get_activations(files, model, batch_size, dims, cuda, verbose=False)
    print(act.shape)
    if pca_path != None:
        pca = PCA(pca_rate)
        pca.fit(act)
        print("pca n components is ")
        print(pca.n_components_)
        pickle.dump(pca, open(pca_path, "wb+"), protocol=4)
      
        act = pca.transform(act)

    pickle.dump(act,open(act_path, "wb+"), protocol=4)
    

def _compute_statistics_of_path(path, model, batch_size, dims, cuda, pca_path, pca_rate, act_path):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        calculate_activation_statistics(files, model, batch_size, dims, cuda, pca_path, pca_rate, act_path)



def calculate_fid_given_paths(paths, batch_size, cuda, dims, pca_path, pca_rate, act_path):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda, pca_path, pca_rate, act_path)




if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    calculate_fid_given_paths(args.path, args.batch_size, args.gpu != '', args.dims, args.pca_path, args.pca_rate, args.act_path)

