import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e1, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold


seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

n_frames = seq.shape[2]

x, y = rect[0], rect[1]
height = rect[3] - rect[1]
width = rect[2] - rect[0]

org_rect = rect
It = seq[:,:,0]
frame1 = It
p_l = np.zeros(2)

rectseq = rect

for i in range(0, n_frames-1):
    
    It1 = seq[:,:,i+1]
    
    p = LucasKanade(It, It1, rect, threshold, num_iters, p_l)
    
    p_new = p + [rect[0]-org_rect[0], rect[1]-org_rect[1]]
    
    p_star = LucasKanade(frame1, It1, org_rect, threshold, num_iters, p_new)
    
    diff = np.linalg.norm(p_new - p_star)
    
    if diff < template_threshold:
        temp = p_star - [rect[0]-org_rect[0], rect[1]-org_rect[1]]
        rect = [rect[0]+temp[0], rect[1]+temp[1], rect[2]+temp[0], rect[3]+temp[1]]
        It = seq[:,:, i+1]
        p_l = np.zeros(2)
        rectseq = np.vstack((rectseq, rect))
        
    else:
        p_l = p
        rectseq = np.vstack((rectseq, [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]))
    print(i)
        
    if i in [0, 99, 199, 299, 399]:
        x,y = rect[0]+p[0],rect[1]+p[1]
        rectangle = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
        fig,ax = plt.subplots(1)
        ax.imshow(It1,cmap='gray')
        ax.add_patch(rectangle)
        plt.axis('off')
        plt.show()
        
        path =  '../results/car_correction_frame_'+str(i+1)+'.png'
        fig.savefig(path, bbox_inches='tight')
        
print(rectseq.shape)
np.save('carseqrects-wcrt.npy', rectseq)