#!/usr/bin/env python3
import os
from functools import partial
from collections import namedtuple
from glob import glob
import numpy as np
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import faulthandler
from memory_profiler import profile

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

Manifold = namedtuple('Manifold', ['features', 'radii_hub', 'hubs'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])
Device= torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda" if torch.cuda.is_available() else "
print(Device)
loops = 0
percent=0

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

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)

    def compute_manifold(self, input):
        # features
        if isinstance(input, str):
            feats = self.extract_features_from_files(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extract_features(input)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extract_features(input)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extract_features_from_files(input)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError

        # radii
        distances = compute_pairwise_distances(feats)

        radii,idxs = distances2radii(distances, k=self.k)
        hubs_list, hubs_value = getHubs(idxs, k=5, level=self.hubsK)
        hubs = feats[hubs_list]
        radii_hub = radii[hubs_list]
        return Manifold(feats, radii_hub, hubs)

    def extract_features(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            print('IPR: resizing %s to (224, 224)' % str((height, width)))
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x):
                return x

        features = []
        for bi in trange(num_batches, desc=desc):
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = images[start:end]
            batch = resize(batch)
            batch = torch.tensor(batch, device=self.device)

            before_fc = self.vgg16.features(batch)
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:6](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def extract_features_from_files(self, path_or_fnames):
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """

        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)
        num_found_images = len(dataloader.dataset)
        desc = 'extracting features of %d images' % num_found_images
        if num_found_images < self.num_samples:
            print('WARNING: num_found_images(%d) < num_samples(%d)' % (num_found_images, self.num_samples))

        features = []
        for batch in tqdm(dataloader, desc=desc):
            # for classifier code
            before_fc = self.vgg16.features(batch.cuda())

            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:6](before_fc)
            features.append(feature.cpu().data.numpy())

        print('done')
        return np.concatenate(features, axis=0)

    def save_ref(self, fname):
        print('saving manifold to', fname, '...')
        #Manifold(feats, radii_hub, hubs)
        np.savez_compressed(fname,
                            feature=self.manifold_ref.features,
                            radii_hub=self.manifold_ref.radii_hub,
                            hubs=self.manifold_ref.hubs)
@profile
def compute_pairwise_distances(X, Y=None, name=None):
    '''
    args:
        X: np.array of shape N x dim
        Y: np.array of shape N x dim
    returns:
        N x N symmetric np.array
    '''

    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
        Y = X
    else:
        num_Y = Y.shape[0]
    print('compute pairwise distances')

    base_size = 35000

    if num_X < base_size:
        base_size = num_X

    row_batch_size = base_size
    col_batch_size = base_size


    batch_predictions = torch.empty([num_X, num_Y], dtype=torch.float16,device='cpu')
    for begin1 in tqdm(range(0, num_Y, row_batch_size)):
        end1 = min(begin1 + row_batch_size, num_Y)
        feature_batch = Y[begin1:end1, :]

        for begin2 in range(0, num_X, col_batch_size):
            end2 = min(begin2 + col_batch_size, num_X)
            ref_batch = X[begin2:end2, :]

            batch_results = sub_distance(ref_batch, feature_batch)  # .tolist()

            batch_predictions[begin2:end2,begin1:end1] = batch_results

    return batch_predictions


def sub_distance(X, Y=None, name=None):
    X=torch.from_numpy(X).to(Device)
    Y=torch.from_numpy(Y).to(Device)

    distances=torch.cdist(X,Y)
    return distances#.detach().cpu().numpy()



def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    idxs=np.zeros(num_features)
    for i in range(num_features):
        idx,radii[i] = get_kth_value(distances[i,:], k=k)
        idx=idx.tolist()
        idxs[idx]=1+idxs[idx]
    return radii,idxs


def get_kth_value(np_array, k):
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    values,idx = torch.topk(np_array.double(), kprime,largest=False)

    # k_smallests = np_array[idx]
    kth_value = values.max()
    return idx,kth_value

def getHubs(idxs, k=5, level=7):
    hub_recorder=idxs
    return np.where(hub_recorder > level)[0], hub_recorder

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
                      glob(os.path.join(root, '**', '*.png'), recursive=True)

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


class FileNames(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)

def to_global(input):
    global loops
    loops=input

def get_custom_loader(image_dir_or_fnames, image_size=224, batch_size=50, num_workers=4, num_samples=-1):
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)

    if isinstance(image_dir_or_fnames, list):
        dataset = FileNames(image_dir_or_fnames, transform)
    elif isinstance(image_dir_or_fnames, str):
        dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError

    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples]
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return data_loader


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, help='Path to the real images')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to use')
    parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    parser.add_argument('--num_samples', type=int, default=50000, help='number of samples to use')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fname_precalc', type=str, default='', help='fname for precalculating manifold')
    parser.add_argument('--featureLoc', type=int, default=30, help='the feature layers')
    parser.add_argument('--hubsK', type=int, default=3, help='hubs k')
    parser.add_argument('--loop_times', type=int, default=0, help='the loops')

    args = parser.parse_args()

    to_global(args.loop_times)
    print('loops: ',loops)

    _root = args.path

    ipr = IPR(batch_size=args.batch_size, k=args.k, num_samples=args.num_samples, featureLoc=args.featureLoc,hubsK=args.hubsK)
    with torch.no_grad():
        # real
        print('compute reals')
        ipr.compute_manifold_ref(_root)
        # save and exit for precalc
        ipr.save_ref(args.fname_precalc)
        print('path_fake (%s) is ignored for precalc' % args.path_fake)


