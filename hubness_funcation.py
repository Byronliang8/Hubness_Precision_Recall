import numpy as np
import torch
# from tqdm import tqdm

Device= torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda" if torch.cuda.is_available() else "

def compute_hub(feats,len):
    distances = compute_pairwise_distances(feats)
    hubs_value = get_hubs(distances, k=7)
    # arrIndex = (np.array(hubs_value).argsort()[::-1]).tolist()
    values, indices = torch.topk(hubs_value, len)

    return values,feats[indices]

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

    batch_predictions = torch.empty([num_X, num_Y], dtype=torch.float16,device=Device)
    for begin1 in (range(0, num_Y, row_batch_size)):
        end1 = min(begin1 + row_batch_size, num_Y)
        feature_batch = Y[begin1:end1, :]

        for begin2 in range(0, num_X, col_batch_size):
            end2 = min(begin2 + col_batch_size, num_X)
            ref_batch = X[begin2:end2, :]

            batch_results = sub_distance(ref_batch, feature_batch)

            batch_predictions[begin2:end2,begin1:end1] = batch_results
        # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
        # If a feature vector is inside a hypersphere of some reference sample, then
        # the new sample lies at the estimated manifold.
        # The radii of the hyperspheres are determined from distances of neighborhood size k.
    return batch_predictions


def sub_distance(X, Y=None):
    num_X = X.shape[0]
    if Y is None:
        num_Y = num_X
        Y = X
    else:
        num_Y = Y.shape[0]
    # X = X.astype(np.float16)  # to prevent underflow
    X=X.to(Device)
    Y=Y.to(Device)

    # distances=torch.cdist(X,Y)
    X_norm_square = torch.sum(X ** 2, axis=1, keepdims=True).detach().cpu().numpy()
    if Y is None:
        Y_norm_square = X_norm_square
    else:
        Y_norm_square = torch.sum(Y ** 2, axis=1, keepdims=True).detach().cpu().numpy()

    X_square = torch.from_numpy(np.repeat(X_norm_square, num_Y, axis=1))
    Y_square = torch.from_numpy(np.repeat(Y_norm_square.T, num_X, axis=0))
    if Y is None:
        Y = X
    # XY = torch.einsum('ij, jk', X, Y.T).to('cpu')
    XY = torch.matmul(X, Y.T).to('cpu')

    diff_square = (X_square - 2 * XY + Y_square)

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = torch.abs(diff_square[idx])  # update to abs
        # print('WARNING: %d negative diff_squares found and set to abs(), min_diff_square=' % idx.sum(),
        #       min_diff_square)

    distances = torch.sqrt(diff_square)
    return distances#.detach().cpu().numpy()

def get_hubs(distances, k=3):
    num_features = distances.shape[0]
    radii = torch.zeros(num_features,device=Device)
    idxs=torch.zeros(num_features,device=Device)
    for i in range(num_features):
        idx,radii[i] = get_kth_value(distances[i,:], k=k)
        idx=idx.tolist()
        idxs[idx]=1+idxs[idx]

    return idxs


def get_kth_value(np_array, k):
    # print(np_array.shape)
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    values,idx = torch.topk(np_array.double(), kprime,largest=False)
    kth_value = values.max()
    return idx,kth_value

# def getHubs(idxs):
#     hub_recorder=idxs
#     return hub_recorder