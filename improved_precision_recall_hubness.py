#!/usr/bin/env python3
from collections import namedtuple
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import faulthandler
from memory_profiler import profile
from getFeature import compute_pairwise_distances

faulthandler.enable()
try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return x

    def trange(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return range(x)

import torch
import torchvision.models as models

Manifold = namedtuple('Manifold', ['features', 'radii_hub', 'hubs'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])
Device= torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda" if torch.cuda.is_available() else "
print(Device)
# loops = 0
# percent=0

class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=10000, featureLoc=-6, hubsK=3, model=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        self.featureLoc = featureLoc
        self.hubsK = hubsK
        self.device = Device#torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None:
            print('loading vgg16 for improved precision and recall...', end='', flush=True)
            self.vgg16 = models.vgg16(pretrained=True).to(device=self.device).eval()
            print('done')
        else:
            self.vgg16 = model

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        # start_time1=time.time()
        manifold_subject = self.compute_manifold(subject)

        precision = compute_metric(self.manifold_ref, manifold_subject.hubs, 'computing precision...')

        recall = compute_metric(manifold_subject, self.manifold_ref.hubs, 'computing recall...')

        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)


    def compute_manifold(self, input):
        '''
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        '''
        # features
        if isinstance(input, str) & input.endswith('.npz'):
            # input is precalculated file
                print('loading', input)
                f = np.load(input)
                feats = f['feature']
                radii_hub=f['radii_hub']
                hubs=f['hubs']
        else:
                print("Warming: Please provide a .npz file")
        return Manifold(feats, radii_hub, hubs)


def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    count = 0

    dist = compute_pairwise_distances(manifold_ref.hubs,feats_subject)

    for i in trange(int(num_subjects)):
        count += (dist[:, i].to(Device) < torch.from_numpy(manifold_ref.radii).to(Device)).any()
    return count / num_subjects

def to_global(input):
    global loops
    loops=input



if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_real', type=str, help='Path to the real images')
    parser.add_argument('path_fake', type=str, help='Path to the fake images')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to use')
    parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples to use')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fname_precalc', type=str, default='', help='fname for precalculating manifold')
    parser.add_argument('--featureLoc', type=int, default=30, help='the feature layers')
    parser.add_argument('--hubsK', type=int, default=3, help='hubs k')
    parser.add_argument('--loop_times', type=int, default=0, help='the loops')

    args = parser.parse_args()

    to_global(args.loop_times)
    print('loops: ',loops)

    r_root = args.path_real
    f_root = args.path_fake

    # Example usage: with real and fake paths
    # python improved_precision_recall.py [path_real] [path_fake]
    ipr = IPR(batch_size=args.batch_size, k=args.k, num_samples=args.num_samples, featureLoc=args.featureLoc,hubsK=args.hubsK)
    with torch.no_grad():

        ipr.compute_manifold_ref(r_root)
        precision, recall = ipr.precision_and_recall(f_root)

    print('precision:', precision)
    print('recall:', recall)

