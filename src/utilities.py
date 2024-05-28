import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from src.caption_dataset import re_eval_dataset
from torch.utils.data import DataLoader
import torch.distributed as dist

def resize_and_crop_image(image, target_width, target_height):
    """
    Resize and crop an image to fit the specified width and height.

    Args:
    image (Image.Image): The original PIL image.
    target_width (int): The target width.
    target_height (int): The target height.

    Returns:
    Image.Image: The resized and cropped image.
    """
    # Calculate the target aspect ratio
    target_ratio = target_width / target_height
    # Calculate the original aspect ratio
    original_ratio = image.width / image.height

    # Determine if the image needs to be cropped by width or height
    if original_ratio > target_ratio:
        # Crop the width
        new_height = image.height
        new_width = int(target_ratio * new_height)
        left = (image.width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = new_height
    else:
        # Crop the height
        new_width = image.width
        new_height = int(new_width / target_ratio)
        left = 0
        top = (image.height - new_height) / 2
        right = new_width
        bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def visualize_meme(image, caption):
    """
    Display a meme image along with its caption.

    Args:
    image (np.array): The image array of the meme.
    caption (str): The caption associated with the meme.
    """
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.imshow(image)  # Show the image
    plt.axis('off')  # Turn off the axis
    plt.title(caption)  # Set the caption as the title
    plt.show()


def recall_at_k(score_matrix, prefix = 't2i_'):
    '''
    Calculating the final R@K scores for image-text retrieval or text-image retrieval.
    The row elements are taken as queries.
    Input:
        score_matrix [torch.tensor]: a matrix 
    Return:
        R@1, R@5, R@10 and R@mean for both image-text retrieval or text-image retrieval.
    '''

    # image indexes
    m_shape = score_matrix.shape
    # top 10 image indexes
    _, rank_txt_idx = score_matrix.topk(10, dim=1)
    # ground truth of image indexes, each row gets extended
    gt_img_j = torch.LongTensor([i for i in range(m_shape[0])]).unsqueeze(1).expand_as(rank_txt_idx)
    # non-zero element indexes
    # nonzero() Return the indices of the elements that are non-zero.
    # rank.shape = (rows, 2), (:, 0) are the values, (:, 1) are the indices 
    rank = (rank_txt_idx == gt_img_j).nonzero()[:, 1]
    if rank.numel():
        r1 = (rank < 1).sum().item() / m_shape[0]
        r5 = (rank < 5).sum().item() / m_shape[0]
        r10 = (rank < 10).sum().item() / m_shape[0]
        r_mean = (r1 + r5 + r10) / 3
    else:
        r1, r5, r10, r_mean = 0, 0, 0, 0
    eval_log = {prefix+'r1': r1,
                prefix+'r5': r5,
                prefix+'r10': r10,
                prefix+'r_mean': r_mean
                }
    return eval_log


def create_dataset(config):
    '''
    Create dataset for ALBEF model

    '''
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   

    test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])    
    print(len(test_dataset))            
    return test_dataset   

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders   
