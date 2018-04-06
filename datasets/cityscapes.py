import numpy as np

def get_colour_map():

    palette = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]]
    palette = np.array(palette).ravel()

    return palette

def get_label_names():

    names = ['road', 'sidewalk', 'building', 'wall',
             'fence', 'pole', 'traffic light', 'traffic sign',
             'vegetation', 'terrain', 'sky', 'person',
             'rider', 'car', 'truck', 'bus', 'train',
             'motorcycle', 'bicycle']

    return names
