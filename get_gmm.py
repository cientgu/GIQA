import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pickle
from sklearn import mixture


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--act_path', type=str, default=None, help='input act path')
parser.add_argument('--kernel_number', type=int, default=5, help='number of gmm kernels')
parser.add_argument('--gmm_path', type=str, default=None, help='output gmm path')



if __name__ == '__main__':
    args = parser.parse_args()

    act = pickle.load(open(args.act_path, "rb"))
    print(act.shape)
    # gact = mixture.GaussianMixture(n_components=args.kernel_number, covariance_type='diag', max_iter=1000)
    print("this is full") 
    gact = mixture.GaussianMixture(n_components=args.kernel_number, covariance_type='full')
    gact.fit(act)
    pickle.dump(gact,open(args.gmm_path, "wb+"))
    
