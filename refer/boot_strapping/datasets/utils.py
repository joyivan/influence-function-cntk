import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.transform import resize


#def preprocessed_dataloader(dataloader, split_size=1):
#    # Load the preprocessed data in lazy way by using dataloader
#    # Split data into split_size and do augmentations
#    # We do this in order to process data efficiently using numpy operations
#
#    assert(type(split_size)==int, 'split_size must be integer')
#
#    batch_size = None
#
#    for X, y in dataloader:
#        # Preprocessing
#        X, y = X.numpy(), y.numpy()
#        
#        if batch_size == None:
#            assert((num_samples % split_size)==0)
#            batch_size = num_samples / split_size
#
#        num_samples = len(X)
#        cursor = 0
#        while True:
#            next_cursor = min(cursor + batch_size, num_samples)
#            if next_cursor == cursor:
#                break
#            else:
#                yield X[cursor:next_cursor], y[cursor:next_cursor]
#            cursor = next_cursor

def view_image_samples(images, n_samples=30, border=2, n_cols=6):
    """
    Show random samples of images, as a form of matrix.
    :param images: np.ndarray, shape: (N, C, H, W).
    :param n_samples: int, number of samples shown.
    :param border: int, border width.
    :param n_cols: int, number of columns in the matrix.
    :return np.ndarray: (N, H, W, C).
    """
    n_rows = math.ceil(n_samples / n_cols)

    N, C, H, W = images.shape
    H += 2*border
    W += 2*border
    if C == 1:
        image_mode = 'L'
    elif C == 3:
        image_mode = 'RGB'
    else:    # nchannel == 4:
        image_mode = 'RGBA'
    image_samples = Image.new(image_mode, (W*n_cols, H*n_rows), color=0)
    draw = ImageDraw.Draw(image_samples)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

    if n_samples > N:
        n_samples = N
    sampled_idxs = np.random.choice(np.arange(images.shape[0], dtype=np.int),
                                    size=n_samples, replace=False)
    idx = 0
    for r in range(n_rows):
        y_pos = r * H
        for c in range(n_cols):
            x_pos = c * W
            if idx >= n_samples: continue
            sample_idx = sampled_idxs[idx]

            image_samples.paste(Image.fromarray((images[sample_idx].transpose(1, 2, 0).squeeze() * 255)\
                                                .astype(np.uint8)), (x_pos+border, y_pos+border))
            draw.text((x_pos+border+10, y_pos+border+10), str(sample_idx), font=fnt, fill=255)

            idx += 1

    return image_samples


def view_image_cam_pairs(images, cams, n_samples=10, border=2, n_cols=6, blend=True):
    """
    Show (image, CAM) pairs of images, as a form of matrix.
    :param images: np.ndarray, shape: (N, C, H, W).
    :param cams: np.ndarray, shape: (N, C, H, W).
    :param n_samples: int, number of samples shown.
    :param border: int, border width.
    :param n_cols: int, number of columns in the matrix.
    :param blend: bool, whether to blend the original image and CAM for each pair.
    :return: np.ndarray, shape: (N, H, W, C).
    """
    N, C, H, W = images.shape
    n_rows = math.ceil(n_samples / n_cols)    # NOTE: to be doubled

    H += 2*border
    W += 2*border
    image_mode = 'RGB'
    image_cam_samples = Image.new(image_mode, (W*n_cols, 2*H*n_rows), color=0)
    draw = ImageDraw.Draw(image_cam_samples)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

    if n_samples > N:
        n_samples = N
    sampled_idxs = np.random.choice(np.arange(images.shape[0], dtype=np.int),
                                    size=n_samples, replace=False)
    idx = 0
    for r in range(n_rows):
        y_pos = 2*r * H
        for c in range(n_cols):
            x_pos = c * W
            if idx >= n_samples: continue
            sample_idx = sampled_idxs[idx]

            # First sub-row: original image
            y_pos_img = y_pos + 0
            image_cam_samples.paste(Image.fromarray((images[sample_idx].transpose(1, 2, 0).squeeze() * 255)\
                                                    .astype(np.uint8)), (x_pos+border, y_pos_img+border))
            draw.text((x_pos+border+10, y_pos_img+border+10), str(sample_idx), font=fnt, fill=255)

            # Second sub-row: CAM image
            y_pos_cam = y_pos + H
            if blend:
                alpha = 0.5
                cam = (alpha * images[sample_idx] + (1-alpha) * cams[sample_idx])
            else:
                cam = cams[sample_idx]
            image_cam_samples.paste(Image.fromarray((cam.transpose(1, 2, 0).squeeze() * 255)\
                                                    .astype(np.uint8)), (x_pos+border, y_pos_cam+border))
            idx += 1

    return image_cam_samples


def view_image_cam_pairs_with_entropy(images, cams, preds, n_samples=10, border=2, n_cols=6, blend=True):
    """
    Show (image, CAM) pairs of images, sorted by entropy in descending order, as a form of matrix.
    :param images: np.ndarray, shape: (N, C, H, W).
    :param cams: np.ndarray, shape: (N, C, H, W).
    :param preds: np.ndarray, shape: (N, num_classes).
    :param n_samples: int, number of samples shown.
    :param border: int, border width.
    :param n_cols: int, number of columns in the matrix.
    :param blend: bool, whether to blend the original image and CAM for each pair.
    :return: np.ndarray, shape: (N, H, W, C).
    """
    N, C, H, W = images.shape
    n_rows = math.ceil(n_samples / n_cols)    # NOTE: to be doubled

    H += 2*border
    W += 2*border
    image_mode = 'RGB'
    image_cam_samples = Image.new(image_mode, (W*n_cols, 2*H*n_rows), color=0)
    draw = ImageDraw.Draw(image_cam_samples)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 30)

    if n_samples > N:
        n_samples = N
    sampled_idxs = np.random.choice(np.arange(images.shape[0], dtype=np.int),
                                    size=n_samples, replace=False)

    # Compute entropies for each examples, and sort indices by entropies
    entropies = -np.sum(np.multiply(preds, np.log(preds)), axis=1)
    sorted_indices = entropies[sampled_idxs].argsort()
    sampled_idxs = sampled_idxs[sorted_indices]

    idx = 0
    for r in range(n_rows):
        y_pos = 2*r * H
        for c in range(n_cols):
            x_pos = c * W
            if idx >= n_samples: continue
            sample_idx = sampled_idxs[idx]

            # First sub-row: original image
            y_pos_img = y_pos + 0
            image_cam_samples.paste(Image.fromarray((images[sample_idx].transpose(1, 2, 0).squeeze() * 255) \
                                                    .astype(np.uint8)), (x_pos+border, y_pos_img+border))
            draw.text((x_pos+border+10, y_pos_img+border+10), str(sample_idx), font=fnt, fill=255)
            draw.text((x_pos+W//2, y_pos_img+H-border-30), '{:.4f}'.format(entropies[sample_idx]),
                      font=fnt, fill=255)

            # Second sub-row: CAM image
            y_pos_cam = y_pos + H
            if blend:
                alpha = 0.5
                cam = (alpha * images[sample_idx] + (1-alpha) * cams[sample_idx])
            else:
                cam = cams[sample_idx]
            image_cam_samples.paste(Image.fromarray((cam.transpose(1, 2, 0).squeeze() * 255) \
                                                    .astype(np.uint8)), (x_pos+border, y_pos_cam+border))
            idx += 1

    return image_cam_samples


def view_tsne_embeddings(embeddings, images, labels):
    """
    Show images as T-SNE embeddings, with the number of embeddings for each point.
    :reference https://cs.stanford.edu/people/karpathy/cnnembed
    :param embeddings: np.ndarray, shape: (N, 2).
    :param images: np.ndarray, shape: (N, C, H, W).
    :param labels: np.ndarray, shape: (N, num_channels).
    :return:
    """
    # Define PASCAL VOC-style color palette
    palette_dict = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0),
                    4: (0, 0, 128), 5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128),
                    8: (64, 0, 0), 9: (192, 0, 0), 10: (64, 128, 0), 11: (192, 128, 0),
                    12: (64, 0, 128), 13: (192, 0, 128), 14: (64, 128, 128), 15: (192, 128, 128),
                    16: (0, 64, 0), 17: (128, 64, 0), 18: (0, 192, 0), 19: (128, 192, 0),
                    20: (0, 64, 128)}

    S = 10000    # a side length of full embedding image
    s = 100      # a side length of every single image
    b = 8       # border width, indicating true class for an image
    image_mode = 'RGB'
    emb_image = Image.new(image_mode, (S, S), color=0)
    draw = ImageDraw.Draw(emb_image)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    # emb_image = np.zeros((S, S, 3), dtype=np.uint8)
    num_embeddings_dict = dict()

    # normalize embeddings: ranged in [0.0, 1.0]
    embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
    # convert one-hot encoded labels to scalar labels
    labels = labels.argmax(axis=1)    # shape: (N,)

    for emb, image, label in zip(embeddings, images, labels):
        # image.shape: (C, H, W), emb.shape: (2,)
        # location for each image
        y = math.ceil(emb[0] * (S-s) + 1)
        x = math.ceil(emb[1] * (S-s) + 1)
        y -= (y-1) % s + 1
        x -= (x-1) % s + 1
        if (y, x) not in num_embeddings_dict:
            num_embeddings_dict[(y, x)] = 0
        num_embeddings_dict[(y, x)] += 1    # increase the number of embeddings in current position

        # color inner border of image, with its symbolic color according to its true class
        #emb_image.paste(Image.fromarray((palette_dict[label]*np.ones((s, s, 3))).astype(np.uint8)), (x, y))
        emb_image.paste(Image.fromarray((palette_dict[0]*np.ones((s, s, 3))).astype(np.uint8)), (x, y))

        # resize and convert image
        image = (resize(image.transpose(1, 2, 0), (s, s), mode='constant') * 255).astype(np.uint8)
        emb_image.paste(Image.fromarray(image[b:-b, b:-b]), (x+b, y+b))

    # draw text, showing the number of overlapping images
    for (y, x), n in num_embeddings_dict.items():
        draw.text((x+b//2, y+b//2), str(n), font=fnt, fill=(255, 255, 255))

    return emb_image

