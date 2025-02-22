import collections

import torchvision.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive
from torchvision import transforms
import os
from PIL import Image
import flgo.benchmark

domain_list = [
    'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
]
class_list = [
    'lollipop', 'apple', 'diamond', 'helmet', 'skull', 'palm_tree', 'lipstick', 'cat', 'rhinoceros', 'peanut',
    'animal_migration', 'pond', 'ant', 'fire_hydrant', 'jacket', 'blueberry', 'microwave', 'remote_control', 'tree', 'paintbrush',
    'butterfly', 'see_saw', 'crown', 'leaf', 'boomerang', 'drill', 'toaster', 'lightning', 'toe', 'garden_hose',
    'sword', 'fork', 'pear', 'hand', 'fireplace', 'sandwich', 'strawberry', 'raccoon', 'bench', 'ice_cream',
    'piano', 'basket', 'chandelier', 'elbow', 'sun', 'cactus', 'car', 'crab', 'cello', 'peas',
    'pig', 'hot_air_balloon', 'tractor', 'hammer', 'ocean', 'canoe', 'screwdriver', 'river', 'feather', 'snail',
    'eye', 'bed', 'violin', 'golf_club', 'tooth', 'diving_board', 'yoga', 'hockey_stick', 'rain', 'cup',
    'calendar', 'stereo', 'radio', 'angel', 'trombone', 'snowman', 'sweater', 'microphone', 'aircraft_carrier', 'calculator',
    'camouflage', 'shovel', 'string_bean', 'television', 'hourglass', 'saw', 'rollerskates', 'bottlecap', 'steak', 'donut',
    'eraser', 'mushroom', 'squiggle', 'stethoscope', 'rifle', 'dog', 'alarm_clock', 'clarinet', 'bee', 'belt',
    'face', 'couch', 'foot', 'spreadsheet', 'dolphin', 'soccer_ball', 'scorpion', 'postcard', 'onion', 'garden',
    'candle', 'speedboat', 'birthday_cake', 'giraffe', 'bear', 'grass', 'flower', 'harp', 'potato', 'bridge',
    'mailbox', 'penguin', 'zebra', 'camera', 'drums', 'underwear', 'swing_set', 'moustache', 'baseball', 'sheep',
    'tennis_racquet', 'square', 'panda', 'mosquito', 'lobster', 'duck', 'cruise_ship', 'shoe', 'moon', 'trumpet',
    'church', 'camel', 'owl', 'tiger', 'rake', 'blackberry', 'lantern', 'firetruck', 'van', 'streetlight',
    'whale', 'stitches', 'power_outlet', 'oven', 'crayon', 'crocodile', 'guitar', 'chair', 'wheel', 'sink',
    'windmill', 'helicopter', 'bus', 'headphones', 'dishwasher', 'triangle', 'dresser', 'The_Great_Wall_of_China', 'picture_frame', 'matches',
    'ladder', 'ceiling_fan', 'nose', 'mouth', 'The_Eiffel_Tower', 'snowflake', 'sailboat', 'key', 'motorbike', 'hexagon',
    'snorkel', 'hot_dog', 'basketball', 'ambulance', 'vase', 'light_bulb', 'zigzag', 'submarine', 'megaphone', 'watermelon',
    'beard', 'passport', 'police_car', 'cell_phone', 'telephone', 'tent', 'mouse', 'ear', 'smiley_face', 'hockey_puck',
    'saxophone', 'pants', 'frying_pan', 'bowtie', 'toilet', 'roller_coaster', 'tornado', 'stove', 'envelope', 'teddy-bear',
    'star', 'hospital', 'pillow', 't-shirt', 'house_plant', 'map', 'truck', 'campfire', 'barn', 'traffic_light',
    'bucket', 'bird', 'parachute', 'wristwatch', 'cooler', 'hot_tub', 'sock', 'shorts', 'line', 'table',
    'waterslide', 'grapes', 'octagon', 'fence', 'skyscraper', 'parrot', 'nail', 'airplane', 'kangaroo', 'skateboard',
    'cloud', 'mug', 'book', 'rainbow', 'leg', 'dragon', 'syringe', 'sleeping_bag', 'suitcase', 'train',
    'jail', 'umbrella', 'house', 'spider', 'coffee_cup', 'binoculars', 'broom', 'brain', 'monkey', 'flashlight',
    'eyeglasses', 'broccoli', 'spoon', 'bread', 'lighthouse', 'circle', 'hat', 'rabbit', 'scissors', 'mermaid',
    'bathtub', 'cookie', 'compass', 'asparagus', 'school_bus', 'bat', 'washing_machine', 'bush', 'fan', 'knee',
    'sea_turtle', 'cannon', 'banana', 'swan', 'octopus', 'beach', 'wine_bottle', 'axe', 'floor_lamp', 'castle',
    'hamburger', 'backpack', 'toothpaste', 'bicycle', 'arm', 'frog', 'laptop', 'mountain', 'paint_can', 'marker',
    'hurricane', 'lighter', 'paper_clip', 'computer', 'wine_glass', 'hedgehog', 'anvil', 'purse', 'pizza', 'flying_saucer',
    'The_Mona_Lisa', 'toothbrush', 'horse', 'stop_sign', 'popsicle', 'pool', 'flamingo', 'fish', 'stairs', 'pineapple',
    'squirrel', 'goatee', 'bracelet', 'finger', 'cow', 'baseball_bat', 'pickup_truck', 'pencil', 'teapot', 'keyboard',
    'cake', 'pliers', 'lion', 'clock', 'bulldozer', 'necklace', 'carrot', 'flip_flops', 'shark', 'door',
    'snake', 'knife', 'elephant', 'bandage', 'dumbbell'
]
path = os.path.join(flgo.benchmark.data_root, 'domainnet')
classes = [
    'bird',
    'feather',
    'headphones',
    'ice_cream',
    'teapot',
    'tiger',
    'whale',
    'windmill',
    'wine_glass',
    'zebra'
]
class DomainDataset(torchvision.datasets.VisionDataset):
    url_temp = {
        "{}.zip": "http://csr.bu.edu/ftp/visda/2019/multi-source/{}.zip",
        "{}_train.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_train.txt",
        "{}_test.txt":"http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_test.txt"
    }
    file_names = []
    def __init__(self, root, domain:str, split='train', classes = None, download:bool=True, transforms=None, transform=None, target_transform=None):
        super(DomainDataset, self).__init__(root, transforms, transform, target_transform)
        self.domain = domain
        if not os.path.exists(os.path.join(root, "{}_{}.txt".format(self.domain, split))):
            download_url("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_{}.txt".format(self.domain, split), root=self.root)
        if not os.path.exists(os.path.join(root, self.domain)):
            if os.path.exists(os.path.join(root, "{}.zip".format(self.domain))):
                try:
                    print(os.path.join(self.root, "{}.zip".format(self.domain)))
                    extract_archive(os.path.join(self.root, "{}.zip".format(self.domain)), self.root, remove_finished=False)
                except Exception as e:
                    print(e)
                    raise FileExistsError('There exists error in download .zipfile')
            else:
                if download==True:
                    self.download_data()
                else:
                    raise FileExistsError('File not exists. Please set download=True the download the raw data of {}'.format(self.domain))
        with open(os.path.join(root, '{}_{}.txt'.format(self.domain, split)), 'r') as inf:
            self.all_images_path = inf.readlines()
        self.all_label_names = [p.split(os.path.sep)[1] for p in self.all_images_path]
        self.label_list = tuple(sorted(list(set(self.all_label_names))))
        self.set_classes(classes)

    def download_data(self):
        for k,v in self.url_temp.items():
            file_name = k.format(self.domain)
            url = v.format(self.domain) if self.domain in ['infograph', 'quickdraw', 'real', 'sketch'] else v.format(f"groundtruth/{self.domain}")
            if file_name.endswith('.zip'):
                download_and_extract_archive(url, self.root, remove_finished=True)
            else:
                download_url(url, self.root)

    def __getitem__(self, item):
        img_path =self.images_path[item]
        label = self.labels[item]
        image = Image.open(img_path)
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        if image.dtype==torch.uint8 or image.dtype==torch.int8:
            image = image/255.0
        return image, label

    def __len__(self):
        return len(self.images_path)

    def set_classes(self, classes):
        if classes is None:
            classes = self.label_list
        self.classes = classes
        tmp_images = []
        tmp_labels = []
        for i,(img, lb) in enumerate(zip(self.all_images_path, self.all_label_names)):
            if lb in self.classes:
                tmp_images.append(os.path.join(self.root, img.strip().split(' ')[0]))
                tmp_labels.append(lb)
        self.images_path = tmp_images
        self.labels = [self.classes.index(lb) for lb in tmp_labels]

class DomainNet(torch.utils.data.ConcatDataset):
    domains = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
    modes = ['domain', 'label']
    def __init__(self, root, split: str = 'all', classes=None, download=True,
                 transform=None, target_transform=None, transforms=None):
        datasets = []  # 数据集逐domain添加
        self.split = split
        self.classes = classes
        self.mode = 'label'
        self.retain = {}

        self.num_classes = len(self.classes)
        self.num_domains = len(self.domains)
        for i, domain in enumerate(self.domains):
            transform = transform[i] if isinstance(transform, list) and len(transform) > i else transform
            target_transform = target_transform[i] if isinstance(target_transform, list) and len(target_transform) > i else target_transform
            transforms = transforms[i] if isinstance(transforms, list) and len(transforms) > i else transforms
            if split != 'all':  # train/test分开划分
                datasets.append(DomainDataset(root, domain=domain, split=split, classes=classes, download=download,
                                              transform=transform, target_transform=target_transform,
                                              transforms=transforms))
            else:
                data_train = DomainDataset(root, domain=domain, split='train', classes=classes, download=download,
                                            transform=transform, target_transform=target_transform,
                                            transforms=transforms)
                data_test = DomainDataset(root, domain=domain, split='test', classes=classes, download=download,
                                           transform=transform, target_transform=target_transform,
                                           transforms=transforms)
                datasets.append(torch.utils.data.ConcatDataset([data_train, data_test]))
        super().__init__(datasets)

        # 构建标签和领域的索引映射
        self.index_map = self._build_index_map()
        print(f"domain distribution: {[[self.sample_count[d, cls] for cls in range(self.num_classes)] for d in range(self.num_domains)]}")
        print(f"class distribution: {[[self.sample_count[d, cls] for d in range(self.num_domains)] for cls in range(self.num_classes)]}")

    def _build_index_map(self):
        """
        构建 index_map: 记录全局索引和 (领域索引, 标签索引) 的映射关系。
        同时统计每个 (领域, 标签) 的样本数量。
        """
        index_map = {}
        self.sample_count = collections.defaultdict(int)  # 用于统计样本数量
        global_idx = 0

        for domain_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                _, label = dataset[local_idx]  # 获取标签
                index_map[global_idx] = (domain_idx, label)
                self.sample_count[(domain_idx, label)] += 1  # 累计 (领域, 标签) 的数量
                global_idx += 1

        return index_map

    def get_idxes(self, mode):
        if mode == 'label':
            idxes = list(range(len(self.classes)))
        elif mode == 'domain':
            idxes = list(range(len(self.domains)))
        else:
            raise ValueError(f"mode must be one of {self.modes}")
        return idxes

    def set_mode(self, mode: str='label'):
        if mode not in self.modes:
            raise ValueError(f"mode must be one of {self.modes}")
        self.mode = mode

    def set_retain(self, retain: dict[str, int | list | tuple]={}):
        """
        设置数据集的访问模式。
        Args:
            mode (str): 'default', 'by_label', 'by_domain' 三种模式之一。
        """

        if type(retain) != dict:
            raise ValueError(f"retain must be dict for {self.modes}")
        for k, v in retain.items():
            if k not in self.modes:
                raise ValueError(f"unknown entry mode：{k}")
            idxes = self.get_idxes(k)
            if type(retain) == int:
                if v < 0:
                    retain[k] = idxes
                elif v not in idxes:
                    raise ValueError(f"retain must be negative or index in range of {k}")
            elif type(v) in (tuple, list):
                for r in v:
                    if r < 0 or r not in idxes:
                        raise ValueError(f"retain must be index in range of {k}")

        self.retain = retain

    def is_retain(self, info):
        for k, v in self.retain.items():
            if k == 'domain' and info[0] not in v:
                return False
            if k == 'label' and info[1] not in v:
                return False
        return True

    def get_global_idx(self, idx):
        """
        输入当前 mode 与 retain 决定下的数据集中样本索引
        """
        local_idx = -1
        for global_idx, info in self.index_map.items():
            if self.is_retain(info):
                local_idx += 1
            if local_idx == idx:
                return global_idx


    def __getitem__(self, idx):
        """
        输入当前 mode 与 retain 决定下的数据集中样本索引
        """
        local_idx = -1
        if self.mode == "label":
            for global_idx, info in self.index_map.items():
                if self.is_retain(info):
                    local_idx += 1
                if local_idx == idx:
                    data, _ = super().__getitem__(global_idx)
                    return data, info[1]

        elif self.mode == "domain":
            for global_idx, info in self.index_map.items():
                if self.is_retain(info):
                    local_idx += 1
                if local_idx == idx:
                    data, _ = super().__getitem__(global_idx)
                    return data, info[0]
        else:
            raise ValueError(f"unknown entry mode：{self.mode}")

    def __iter__(self):
        """
        返回 mode 规定的属性，并过滤 retain 之外的数据。
        """
        # local_idx = 0
        if self.mode == "label":
            for global_idx, info in self.index_map.items():
                if self.is_retain(info):
                    data, _ = super().__getitem__(global_idx)
                    yield data, info[1]
                    # local_idx += 1

        elif self.mode == "domain":
            for global_idx, info in self.index_map.items():
                if self.is_retain(info):
                    data, _ = super().__getitem__(global_idx)
                    yield data, info[0]
                    # local_idx += 1
        else:
            raise ValueError(f"unknown entry mode：{self.mode}")

    def __len__(self):
        return len([None for info in self.index_map.values() if self.is_retain(info)])

    def set_classes(self, classes):
        if self.split == 'all':
            for dataset in self.datasets:
                for ds in dataset.datasets:
                    ds.set_classes(classes)
        else:
            for dataset in self.datasets:
                dataset.set_classes(classes)
        self.classes = classes
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.domain_ids = []
        g = 0
        for i in range(len(self.datasets)):
            if i>=self.cumulative_sizes[g]:
                g += 1
            self.domain_ids.append(g)

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.PILToTensor(),
])
train_data = DomainNet(path, split='train', classes=classes, transform=transform)
test_data = DomainNet(path, split='test', classes=classes, transform=transform)
import torch.nn as nn
from collections import OrderedDict

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024,1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn6(self.fc1(x))
        x = self.relu(x)
        x = self.bn7(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        return x

def get_model():
    # return torchvision.models.get_model('resnet18', num_classes = len(classes))
    return AlexNet(len(train_data.classes))