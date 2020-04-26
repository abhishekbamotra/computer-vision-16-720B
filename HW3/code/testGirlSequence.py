import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

n_frames = seq.shape[2]

x, y = rect[0], rect[1]
height = rect[3] - rect[1]
width = rect[2] - rect[0]

rectseq = rect

for i in range(0, n_frames-1):
    p = LucasKanade(seq[:,:,i], seq[:,:,i+1], rect, threshold, num_iters)
    print(i)
    
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    
    rectseq = np.vstack((rectseq, rect))
    
    if i in [0, 19, 39, 59, 79]:
        x,y = rect[0],rect[1]
        rectangle = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
        fig,ax = plt.subplots(1)
        ax.imshow(seq[:,:,i+1],cmap='gray')
        ax.add_patch(rectangle)
        plt.axis('off')
        plt.show()
        path =  '../results/girlseq_frame_'+str(i+1)+'.png'
        fig.savefig(path, bbox_inches='tight')
        
print(rectseq.shape)
np.save('girlseqrects.npy', rectseq)