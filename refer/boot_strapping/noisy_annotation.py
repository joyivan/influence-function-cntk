import glob
import json
import numpy as np
from ipdb import set_trace
import time

def noisy_annotation(p):
    # save annotation with noisy label
    file_type = 'balanced'
    
    # annotation (file_type)
    dic = {'classes': {}, 'images':{}}
    tmp = {}
    
    directory = '../{}/original/image/'.format(file_type)
    list_image = glob.glob(directory+'*.png')
    
    # extract classes
    classes = []
    for ls in list_image:
        # extract class info append to list_image
        fname = ls.split('/')[-1]
        classes.append(fname.split('_')[1])
    
    # find unique elementsi (for sanity check)
    classes_unq = list(set(classes))
    print(classes_unq)
    print('num of classes:', len(classes_unq))
    
    with open('../raw/emnist-{}-mapping.txt'.format(file_type)) as fid:
        mp = fid.read().split('\n')[:-1]
    mapping = {int(x.split(' ')[0]): chr(int(x.split(' ')[1])) for x in mp}
    dic['classes'] = mapping
    num_classes = len(mapping)
    print('num of classes:', num_classes)
    
    inv_mapping = {val:key for key,val in mapping.items()}
    
    # add image metadata
    for f in list_image:
        fname = f.split('/')[-1]
        c = fname.split('_')[1]
        #npy = np.asarray(Image.open(f))
        cls = int(inv_mapping[c])
        cls_hat = class_flip(int(cls), num_classes, p)
        dic['images'][fname] = {
                'size':{
                    'height': 28, #npy.shape[0],
                    'width': 28, #npy.shape[1],
                    'channel': 1 # 1 if len(npy.shape)==2 else npy.shape[2]
                    },
                'class': [cls_hat]
                }
        #print(int(inv_mapping[c]))
    
    # write
    h = json.dumps(dic, sort_keys=True, indent=4, separators=(',',':'), ensure_ascii=False)
    with open('../{}/original/annotation/annotation1_wp_{}.json'.format(file_type,p), 'w') as f:
        f.write(h)

def class_flip(cls, num_classes, p):
    # cls: 1 dimension integer class
    # num_classes: number of classes
    # p: probability p (0 <= p <= 1)
    assert(p<=1 and p>=0)

    rand = np.random.rand(1)[0]
    if rand < 1-p:
        # with probability 1-p, class information is conserved
        return cls
    else:
        # with probability p, class is flipped uniformly
        idx = np.random.randint(num_classes-1)
        clss = list(range(num_classes))
        clss.remove(cls)
        return clss[idx]

set_trace()
num = 20
t1 = time.time()
for i in range(num):
    p = i /num
    noisy_annotation(p)
print(time.time()-t1, 'time elapsed')
