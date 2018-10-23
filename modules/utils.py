# utils
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# show_image_from_data
# visualize_topk_samples
# tsne?

def show_image_from_data(img, show=True):
    # show image from numpy array
    # img: (C,W,H) numpy array

    if show:
        return 0

    #img_show = np.squeeze(np.transpose(img,[1,2,0]))
    img_show = np.transpose(img,[1,2,0])
    imshow(img_show)
    plt.show()

    return 0

def visualize_topk_samples(measure, data_set, num_sample=5, mask=None,\
        verbose='A/D', show=False, save_path='/Data/result/'):
    # measure: list of measure whose each element represents score of each datapoints
    # data_set: set of data to be visualized (e.g. train_set)
    # num_sample: the number of samples to be visualized
    # mask: (0,1) list for training set
    # verbose:
    #   ALL: show DISADV, ADV, INF, NEG examples
    #   A/D: show advantageous and disadvantageous examples
    # show: indicator that chooses plt.show or not
    
    def extract_annotation(data_set, indices, **kwargs):
        # extract image annotations from dataset w.r.t. indices
        
        # data_set: dataset structure
        # indices: set of sorted and sampled index (e.g. topk)
        # kwargs: other information to be annotated
        #   key: name of this feature (str) (e.g. influence function)
        #   value: value set of this feature for each datapoints (N x 1 numpy array) (e.g. if value)

        images = []; annotations = []

        for idx in indices:
            img, lb = data_set.__getitem__(idx)
            lb_str = data_set.anno_dict['classes'][str(np.argmax(lb))]
            filename = data_set.filename_list[idx]
            
            annotation = [\
                    'training set name: {}'.format(filename),\
                    'training set label(anno_dict): {}'.format(lb_str),\
                    ]

            for key in kwargs.keys():
                annotation.append('{}: {}'.format(key, kwargs[key][idx]))

            images.append(img)
            annotations.append('\n'.join(annotation))

        images = np.array(images)

        return images, annotations

    def draw_images_with_titles(images, filenames, show=False,\
            save_dir='/Data/result/images_with_titles.png'):
        
        N, Ch, H, W = images.shape

        if Ch == 1:
            images = np.tile(images, (1,3,1,1))

        fig, axes = plt.subplots(N, 1, figsize=(H,W))

        for idx in range(N):
            image = images[idx].transpose((1,2,0))
            filename = filenames[idx]
            _ = axes[idx].imshow(image)
            _ = axes[idx].axis('off')
            _ = axes[idx].set_title(filename)

        plt.savefig(os.path.join(save_dir))
        
        if show:
            plt.show()

        return 0
    
    num_data = data_set.__len__()
    assert(len(measure)==num_data) 

    if mask == None:
        argsort = np.argsort(measure)
    else:
        assert(len(mask) == len(measure))
        argsort = list(filter(lambda idx: mask[idx], np.argsort(measure)))

    topk = argsort[-1:-num_sample-1:-1] # samples that increase loss a lot
    botk = argsort[0:num_sample] # samples that decrease loss a lot

    # make folder
    if os.path.exists(save_path):
        os.makedirs(save_path)

    # visualize advantageous and disadvantageous samples
    if verbose == 'A/D' or verbose == 'ALL':
        # advantageous samples
        images, annotations = extract_annotation(data_set, topk, **{'measure': measure})
        draw_images_with_titles(images, annotations, show=show,\
                save_dir=save_path+'_disadvantageous.png')
        # disadvantageous samples
        images, annotations = extract_annotation(data_set, botk, **{'measure': measure})
        draw_images_with_titles(images, annotations, show=show,\
                save_dir=save_path+'_advantageous.png')
    
    # visualize influential and negligible samples
    if verbose == 'ALL':
        # absolutize and masking
        measure_abs = np.abs(measure)
        if mask == None:
            argsort = np.argsort(measure_abs)
        else:
            assert(len(mask) == len(measure_abs))
            argsort = list(filter(lambda idx: mask[idx], np.argsort(measure_abs)))

        topk = argsort[-1:-num_sample-1:-1]
        botk = argsort[0:num_sample]
        # influential samples
        images, annotations = extract_annotation(data_set, topk, **{'measure': measure})
        draw_images_with_titles(images, annotations, show=show,\
                save_dir=save_path+'_influential.png')
        # negligible samples
        images, annotations = extract_annotation(data_set, botk, **{'measure': measure})
        draw_images_with_titles(images, annotations, show=show,\
                save_dir=save_path+'_negligible.png')

    return 0

