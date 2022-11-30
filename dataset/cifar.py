# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from collections import Counter


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, bcl=False, mixup=0):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        self.transform = transform
        self.bcl = bcl
        self.mixup = mixup
        self.num_samples = len(self.data)

        # set of most imbalanced classes (Classes with reasonably less samples than expected for uniform distribution)
        self.least_freq_cls = set([c for c, count in Counter(self.targets).most_common() if (count + (self.num_samples//(len(img_num_list)*1.5))) < (self.num_samples//len(img_num_list))])
        self.least_freq_indices = self.get_low_freq_idx(self.least_freq_cls)
        
        if bcl and train:
            assert len(transform) >= 3, "Please provide a list of 3 tranforms for bcl training"

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_low_freq_idx(self, classes):
        low_freq = []
        for i, t in enumerate(self.targets):
            if t in classes:
                low_freq.append(i)
        return low_freq
    
    def __getitem__(self, index):
        sample = Image.fromarray(self.data[index], mode="RGB")
        label = self.targets[index]
        if self.transform is not None:
            if self.train and self.bcl:
                # Simplfy transforms for speedup
                sample1 = self.transform[0](sample)
                # sample2 = torch.clone(sample1)
                # sample3 = torch.clone(sample1)
                sample2 = self.transform[1](sample)
                sample3 = self.transform[2](sample)

                mixup_label = label #torch.nn.functional.one_hot(torch.tensor([label]), self.cls_num)
                
                # Referencing https://gist.github.com/ttchengab/49bfe3af8ab76561f1db107adc953b53#file-mixupdata-py
                if self.mixup > 0 and random.random() < 0.3:
                    rand_idx = random.randint(0, len(self.least_freq_indices)-1)
                    rand_sample = self.transform[0](Image.fromarray(self.data[self.least_freq_indices[rand_idx]], mode="RGB"))
                    rand_target = torch.nn.functional.one_hot(torch.tensor([self.targets[rand_idx]]), self.cls_num)

                    lam = np.random.beta(self.mixup, self.mixup)
                    sample1 = lam * sample1 + (1 - lam) * rand_sample
                    # sample2 = lam * sample2 + (1 - lam) * rand_sample
                    # sample3 = lam * sample3 + (1 - lam) * rand_sample
                    mixup_label = lam * mixup_label + (1 - lam) * rand_target

                return [sample1, sample2, sample3], label, mixup_label
            else:
                return self.transform(sample), label


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='/DATACENTER/3/zjg/cifar', train=True,
                                 download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb;

    pdb.set_trace()