from miso.object_detection.engine import transforms as T


def get_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomTranspose(0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort())
        pass
    return T.Compose(transforms)
