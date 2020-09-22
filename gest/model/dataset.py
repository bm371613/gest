import json
import pathlib

from PIL import Image
from torchvision.datasets import VisionDataset


def make_dataset(directory, class_to_idx):
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = directory / target_class
        if not target_dir.is_dir():
            continue
        for path in sorted(target_dir.glob('*.jpg')):
            additional_annotations = None
            additional_annotations_path = path.with_suffix('.json')
            if additional_annotations_path.exists():
                additional_annotations = json.load(additional_annotations_path.open())
            target = (class_index, additional_annotations)
            instances.append((path, target))
    return instances


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Dataset(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None):
        root = pathlib.Path(root).expanduser()
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = pil_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in dir.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
