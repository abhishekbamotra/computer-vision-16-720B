import argparse
import numpy as np
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion


# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

n_frames = seq.shape[2]
count = 0
for i in range(0, n_frames-1) :
    print(i)
    frame = np.zeros([seq[:,:,i].shape[0], seq[:,:,i+1].shape[1], 3])
    mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
    pts = np.where(mask == 0)

    if i in [29, 59, 89, 119]:
        fig = plt.figure()
        plt.imshow(seq[:,:,i+1], cmap='gray')
        plt.axis('off')
        plt.plot(pts[1], pts[0], 'x', color='r')
        path = str('../results/aerial' + str(i+1) + '.png')
        fig.savefig(path, bbox_inches='tight')
        
        