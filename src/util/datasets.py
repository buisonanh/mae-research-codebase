import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    if hasattr(args, 'dataset_type') and args.dataset_type == 'rafdb':
        # For RAF-DB dataset
        if is_train:
            root = args.data_path
        else:
            # Assuming validation set is in a 'test' folder for RAF-DB
            root = args.data_path.replace('train', 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        # Original ImageNet structure
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    
    # Check if we're using RAF-DB dataset
    if hasattr(args, 'dataset_type') and args.dataset_type == 'rafdb':
        # Facial expression datasets often benefit from different augmentations
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),  # Less aggressive crop for faces
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color jitter
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            return transform
        else:
            # Eval transform for RAF-DB
            t = []
            t.append(transforms.Resize((args.input_size, args.input_size)))  # Square resize for faces
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            return transforms.Compose(t)
    
    # Original ImageNet transforms
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            scale=(0.2, 1.0),
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter if hasattr(args, 'color_jitter') else None,
            auto_augment=args.aa if hasattr(args, 'aa') else None,
            interpolation=args.interpolation if hasattr(args, 'interpolation') else 'bicubic',
            re_prob=args.reprob if hasattr(args, 'reprob') else 0,
            re_mode=args.remode if hasattr(args, 'remode') else None,
            re_count=args.recount if hasattr(args, 'recount') else None,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    size = 292
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BILINEAR if hasattr(args, 'interpolation') and args.interpolation == 'bilinear' else
                          PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
