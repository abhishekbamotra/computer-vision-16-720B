import numpy as np

def shiftFrame(frame1, frame2, shift):
    shift_i, shift_j = 0, 0
    min_diff = np.sum((frame1 - frame2)**2)
    
    for i in range(-shift, shift):
        for j in range(-shift, shift):
            
            copy_frame2 = np.roll(frame2, i, axis=0)
            copy_frame2 = np.roll(copy_frame2, j, axis=1)
            
            rolled_diff = np.sum((frame1 - copy_frame2)**2)
            
            if rolled_diff < min_diff:
                min_diff = rolled_diff
                shift_i, shift_j = i, j
 
    frame2 = np.roll(frame2, shift_i, axis=0)
    frame2 = np.roll(frame2, shift_j, axis=1)
    return frame2


def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    
    shift = 30 # max shift 
    
    # Fixed red as the base channel and positioned other channels to align
    # with red channel.
    
    shifted_green = shiftFrame(red, green, shift)
    shifted_blue = shiftFrame(red, blue, shift)
    rgb = np.dstack((red, shifted_green, shifted_blue))

    return rgb
