
from torchvision import transforms

# ImageNet Normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training: Augmentation + Normalization
train_transform = transforms.Compose([

    transforms.Resize((162, 300)),
    transforms.RandomRotation(degrees=5),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# Validation: Nur Normalisierung
val_transform = transforms.Compose([

    transforms.Resize((162, 300)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# Test: Nur Normalisierung
test_transform = transforms.Compose([

    transforms.Resize((162, 300)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])



from torchvision import transforms
from PIL import Image, ImageOps


# Resize while preserving aspect ratio and adding padding
class ResizeWithPadding:

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):

        image = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0)
        )

        x = (self.size - image.width) // 2
        y = (self.size - image.height) // 2

        canvas.paste(image, (x, y))

        return canvas


train_transform_resnet = transforms.Compose([

    transforms.RandomRotation(degrees=5),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02)
    ),

    ResizeWithPadding(224),

    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


val_transform_resnet = transforms.Compose([

    ResizeWithPadding(224),

    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


test_transform_resnet = transforms.Compose([

    ResizeWithPadding(224),

    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
