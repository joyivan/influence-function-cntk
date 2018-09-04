import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import numpy as np
import time

import os, sys
sys.path.append('../refer/boot_strapping')
import json

from skimage.io import imread, imsave

from datasets import dataset28 as dataset

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader

from ipdb import set_trace

def show_image_from_data(img):
    # show image from dataset
    # img: (C,W,H) numpy array
    img_show = np.squeeze(np.transpose(img, [1,2,0]))
    imshow(img_show)
    plt.show()

set_trace()

# FIXME
root_dir = '/Data/emnist/balanced/original'
file_dir = './compare/result_net_nn_if_nn/train_e_99502'
filename = 'if_cg_logreg'
#filename = 'if_se_logreg'

# sample size
#trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=100)
#test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)
trainval_list = np.load(file_dir+'/trainval_list.npy')
test_list = np.load(file_dir+'/test_list.npy')
#temp_list, _ = dataset.read_data_subset(root_dir, mode='test') # 18000 samples
#temp_list, _ = dataset.read_data_subset(root_dir, mode='test', sample_size=2000)
temp_list = list(np.load('./images/tsne/temp_list.npy'))

with open('/Data/emnist/balanced/original/annotation/annotation1.json','r') as fid:
    anno_dict = json.load(fid)
#with open('/Data/emnist/balanced/original/annotation/annotation1_wp_0.3.json','r') as fid:
#    noisy_anno_dict = json.load(fid)
    
train_set = dataset.LazyDataset(root_dir, trainval_list, anno_dict)
test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)
    
# emnist dataset: SANITY CHECK
print(len(test_set), type(test_set))
print(len(test_list))

# emnist network
from models.nn import VGG as ConvNet

hp_d = dict() # hyperparameters for a network
net = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_0.0.dnn')

# emnist network: SANITY CHECK
start_time = time.time()
ys, y_preds, test_score, confusion_matrix = net.predict(test_set, **hp_d)
total_time = time.time() - start_time

print('Test error rate: {}'.format(test_score))
print('Total tack time(sec): {}'.format(total_time))
print('Tact time per image(sec): {}'.format(total_time / len(test_list)))
print('Confusion matrix: \n{}'.format(confusion_matrix))

# t-SNE check

# IF_cg_logreg = IF_val(net, ihvp_cg_logreg, train_set)
# visualize_topk_samples(IF_cg_logreg, num_sample=5)
# np.save('./IF_cg_logreg.npy', IF_cg_logreg)
IF = np.load(file_dir+'/'+filename+'.npy')

num_sample=5
argsort = np.argsort(IF) 
topk = argsort[-1:-num_sample-1:-1]
botk = argsort[0:num_sample]

topk_list = [trainval_list[i] for i in topk]
botk_list = [trainval_list[i] for i in botk]
print(botk_list)

set_trace()
test_idx = np.argmax([file_dir.split('/')[-1] in e for e in test_list])

tsne_list = np.concatenate([test_list, temp_list, topk_list, botk_list, [test_list[test_idx]]]) # list concatenation
tsne_set = dataset.LazyDataset(root_dir, tsne_list, anno_dict)

img_test, lb_test = test_set.__getitem__(test_idx)
show_image_from_data(img_test)

print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))])

from models.utils import grad_cam, get_feature_maps
from sklearn.manifold import TSNE

tsne_feature_maps, images, test_labels = get_feature_maps(net, tsne_set, 'conv3_relu', **hp_d)
print(tsne_feature_maps.shape)

# Flatten feature maps into feature vectors
tsne_features = tsne_feature_maps.reshape((tsne_feature_maps.shape[0], -1))
print(tsne_features.shape)

tic = time.time()
tsne_embeddings = TSNE(n_components=2, verbose=1).fit_transform(tsne_features)
toc = time.time()
print('TSNE takes {} secs'.format(toc-tic))
print(tsne_embeddings.shape)
np.save('./images/tsne/tsne_embeddings.npy',tsne_embeddings)
#tsne_embeddings = np.load('./images/tsne/tsne_embeddings.npy')

from datasets.utils import view_tsne_embeddings

images = np.tile(images,(1,3,1,1))

images[-1][0] = 1 # test image to red
show_image_from_data(images[-1])
for i in range(num_sample):
    #images[-num_sample+i-1] = 0.5 * g + images[-num_sample+i-1] # botk
    #images[-2*num_sample+i-1] = 0.5 * b + images[-2*num_sample+i-1] # topk
    images[-num_sample+i-1][1] = 1  # botk to green
    images[-2*num_sample+i-1][2] = 1 # topk to blue

emb_image = view_tsne_embeddings(tsne_embeddings, images, test_labels)
imsave(os.path.join('./images/tsne/convnet_embed_'+filename+'.jpg'), emb_image)
