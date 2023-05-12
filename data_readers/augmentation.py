import torch
import torchvision.transforms as transforms


class ColorAugmentor:
    def __init__(self):
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5 / 3.14),
            transforms.ToTensor()])

    def color_transform(self, image1, image2):
        """ Perform same perturbation over all images """
        wd = image1.shape[-1]
        image_stack = torch.cat([image1, image2], -1)
        image_stack = 255 * self.augcolor(image_stack / 255.0)
        return image_stack.split([wd, wd], -1)

    def __call__(self, image1, image2):
        image1, image2 = self.color_transform(image1, image2)
        return image1, image2
