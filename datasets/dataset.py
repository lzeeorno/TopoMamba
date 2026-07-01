from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import cv2
from sklearn.model_selection import KFold

def nnunet_light_ct_normalize(image, clip_range=(-125.0, 275.0)):
    """Apply lightweight nnU-Net-style CT windowing only to raw HU-like arrays."""
    image = image.astype(np.float32, copy=False)
    if image.size == 0:
        return image
    finite = np.isfinite(image)
    if not finite.all():
        image = np.nan_to_num(image, nan=0.0, posinf=clip_range[1], neginf=clip_range[0])
    # Already-normalized Synapse npz files are usually near [0, 1] or z-scored.
    if image.min() >= -20.0 and image.max() <= 20.0:
        return image
    lo, hi = clip_range
    image = np.clip(image, lo, hi)
    return (image - lo) / max(hi - lo, 1e-6)


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为tensor并处理通道数
        image = torch.from_numpy(image.astype(np.float32))
        
        # 将单通道图像复制为3通道图像（为了匹配预训练权重）
        # 这样做可以让1通道的医学图像兼容3通道的预训练权重
        image = image.unsqueeze(0).repeat(3, 1, 1)  # 从 (H, W) -> (1, H, W) -> (3, H, W)
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = nnunet_light_ct_normalize(image)
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            image = nnunet_light_ct_normalize(image)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


def create_cvc_five_fold_splits(base_dir):
    """创建CVC-ClinicDB数据集的五折交叉验证划分"""
    
    # 获取图像列表 - 修改为PNG格式
    img_dir = os.path.join(base_dir, 'images')
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]  # 修改为.png
    img_files.sort()
    print(f"📊 总共找到 {len(img_files)} 张PNG图像")
    
    # 创建五折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 创建保存目录
    splits_dir = os.path.join(base_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(img_files)):
        train_files = [img_files[i] for i in train_idx]
        val_files = [img_files[i] for i in val_idx]
        
        # 保存训练集列表
        train_file = os.path.join(splits_dir, f'fold_{fold}_train.txt')
        with open(train_file, 'w') as f:
            for filename in train_files:
                # 只保存不带扩展名的文件名
                basename = os.path.splitext(filename)[0]
                f.write(f"{basename}\n")
        
        # 保存验证集列表
        val_file = os.path.join(splits_dir, f'fold_{fold}_val.txt')
        with open(val_file, 'w') as f:
            for filename in val_files:
                basename = os.path.splitext(filename)[0]
                f.write(f"{basename}\n")
        
        print(f"✅ Fold {fold}: 训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
    
    print(f"🎯 五折交叉验证划分完成，保存在: {splits_dir}")
    return splits_dir


class CVC_dataset(Dataset):
    def __init__(self, base_dir, fold=0, split='train', transform=None):
        """
        CVC-ClinicDB数据集类
        Args:
            base_dir: 数据集根目录
            fold: 折数 (0-4)
            split: 'train' 或 'val'
            transform: 数据变换
        """
        self.base_dir = base_dir
        self.fold = fold
        self.split = split
        self.transform = transform
        
        # 确定图像和掩码目录
        self.img_dir = os.path.join(base_dir, 'images')
        self.mask_dir = os.path.join(base_dir, 'masks')
        
        # 文件扩展名 - 修改为PNG格式
        self.img_ext = '.png'  # 修改为.png
        self.mask_ext = '.png'  # 修改为.png
        
        # 加载五折交叉验证的文件列表
        splits_dir = os.path.join(base_dir, 'splits')
        if not os.path.exists(splits_dir):
            print(f"📁 创建五折交叉验证划分...")
            create_cvc_five_fold_splits(base_dir)
        
        split_file = os.path.join(splits_dir, f'fold_{fold}_{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"找不到划分文件: {split_file}")
        
        with open(split_file, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
        print(f"📊 加载 CVC-ClinicDB Fold {fold} {split} 集: {len(self.sample_list)} 个样本")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        
        # 构建文件路径
        img_path = os.path.join(self.img_dir, sample_name + self.img_ext)
        mask_path = os.path.join(self.mask_dir, sample_name + self.mask_ext)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到掩码文件: {mask_path}")
        
        # 加载图像和掩码 (PNG格式)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 归一化掩码到0-1
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.uint8)  # 二值化
        
        sample = {
            'image': image,
            'label': mask,
            'case_name': sample_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class CVC_RandomGenerator(object):
    """CVC数据集的随机数据增强器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        # 数据增强
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        if random.random() > 0.5:
            image = self.random_color_jitter(image)
        
        # 调整大小
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {
            'image': image, 
            'label': label.long(),
            'case_name': case_name
        }
        return sample
    
    def random_rot_flip(self, image, label):
        """随机旋转和翻转"""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        
        return image, label
    
    def random_rotate(self, image, label):
        """随机旋转"""
        angle = np.random.randint(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image, label
    
    def random_color_jitter(self, image):
        """随机颜色抖动"""
        if random.random() > 0.5:
            # 亮度调整
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255)
        
        if random.random() > 0.5:
            # 对比度调整
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 255)
        
        return image.astype(np.uint8)


class CVC_TestGenerator(object):
    """CVC数据集的测试数据处理器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        # 调整大小
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {
            'image': image, 
            'label': label.long(),
            'case_name': case_name
        }
        return sample
        

class SwinUnet_RandomGenerator(object):
    """
    专门为SwinUNet设计的数据增强器
    与原始Swin-UNet保持一致的数据预处理方式
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 数据增强
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        
        # 调整大小
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为tensor格式 - 与原始Swin-UNet保持一致
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H, W) -> (1, H, W)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


class SwinUnet_Synapse_dataset(Dataset):
    """
    专门为SwinUNet设计的Synapse数据集类
    保持与原始Swin-UNet数据处理方式的完全一致
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split in ["train", "val"] or self.sample_list[idx].strip('\n').split(",")[0].endswith(".npz"):
            slice_name = self.sample_list[idx].strip('\n').split(",")[0]
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            try:
                image, label = data['image'], data['label']
            except:
                image, label = data['data'], data['seg']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class SwinUnet_TestGenerator(object):
    """
    SwinUNet测试时的数据预处理器
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # 调整大小
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 转换为tensor格式
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H, W) -> (1, H, W)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample
        

class ISIC_dataset(Dataset):
    def __init__(self, base_dir, split='train', dataset_type='isic2017', transform=None):
        """
        ISIC数据集类
        Args:
            base_dir: 数据集根目录 (如 './data/isic2017/' 或 './data/isic2018/')
            split: 'train', 'val', 'test' - 使用官方划分
            dataset_type: 'isic2017' 或 'isic2018'
            transform: 数据变换
        """
        self.base_dir = base_dir
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        
        # 根据数据集类型和分割确定目录路径
        if dataset_type == 'isic2017':
            if split == 'train':
                self.img_dir = os.path.join(base_dir, 'isic2017_Train_input_2000/ISIC-2017_Training_Data')
                self.mask_dir = os.path.join(base_dir, 'isic2017_train_mask_2000/ISIC-2017_Training_Part1_GroundTruth')
            elif split == 'val':
                self.img_dir = os.path.join(base_dir, 'isic2017_Validation_input_150/ISIC-2017_Validation_Data')
                self.mask_dir = os.path.join(base_dir, 'isic2017_Validation_mask_150/ISIC-2017_Validation_Part1_GroundTruth')
            elif split == 'test':
                self.img_dir = os.path.join(base_dir, 'isic2017_Test_input_600/ISIC-2017_Test_v2_Data')
                self.mask_dir = os.path.join(base_dir, 'isic2017_Test_mask_600/ISIC-2017_Test_v2_Part1_GroundTruth')
            
            self.img_ext = '.jpg'
            self.mask_ext = '_segmentation.png'
            
        elif dataset_type == 'isic2018':
            if split == 'train':
                self.img_dir = os.path.join(base_dir, 'isic2018_Train_Input_2594')
                self.mask_dir = os.path.join(base_dir, 'isic2018_Train_mask_2594')
            elif split == 'val':
                self.img_dir = os.path.join(base_dir, 'isic2018_Val_Input_100')
                self.mask_dir = os.path.join(base_dir, 'isic2018_Val_mask_100')
            elif split == 'test':
                self.img_dir = os.path.join(base_dir, 'isic2018_Test_Input_1000')
                self.mask_dir = os.path.join(base_dir, 'isic2018_Test_mask_1000')
            
            self.img_ext = '.jpg'
            self.mask_ext = '_segmentation.png'
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        # 检查目录是否存在
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
        
        # 获取所有可用的图像文件名（不包括扩展名）
        all_img_files = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith(self.img_ext):
                # 提取基本文件名（去除扩展名）
                basename = os.path.splitext(filename)[0]
                # 对于有_superpixels的文件，跳过
                if not basename.endswith('_superpixels'):
                    all_img_files.append(basename)
        
        all_img_files.sort()
        self.sample_list = all_img_files
        
        print(f"📊 加载 {dataset_type} {split} 集: {len(self.sample_list)} 个样本")
        print(f"图像目录: {self.img_dir}")
        print(f"掩码目录: {self.mask_dir}")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        
        # 构建文件路径
        img_path = os.path.join(self.img_dir, sample_name + self.img_ext)
        mask_path = os.path.join(self.mask_dir, sample_name + self.mask_ext)
        
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        
        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到掩码文件: {mask_path}")
        
        # 加载图像和掩码
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载图像文件: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码文件: {mask_path}")
        
        # 归一化掩码到0-1
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.uint8)  # 二值化
        
        sample = {
            'image': image,
            'label': mask,
            'case_name': sample_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ISIC_RandomGenerator(object):
    """ISIC数据集的随机数据增强器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        # 数据增强
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        if random.random() > 0.5:
            image = self.random_color_jitter(image)
        
        # 调整大小
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {
            'image': image, 
            'label': label.long(),
            'case_name': case_name
        }
        return sample
    
    def random_rot_flip(self, image, label):
        """随机旋转和翻转"""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        
        return image, label
    
    def random_rotate(self, image, label):
        """随机旋转"""
        angle = np.random.randint(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image, label
    
    def random_color_jitter(self, image):
        """随机颜色抖动"""
        if random.random() > 0.5:
            # 亮度调整
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255)
        
        if random.random() > 0.5:
            # 对比度调整
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 255)
        
        return image.astype(np.uint8)


class ISIC_TestGenerator(object):
    """ISIC数据集的测试数据处理器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        # 调整大小
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {
            'image': image, 
            'label': label.long(),
            'case_name': case_name
        }
        return sample


# ============================================================
# Massachusetts Roads 数据集
# ============================================================

class Roads_dataset(Dataset):
    """
    Massachusetts Roads 卫星道路分割数据集
    
    目录结构:
      massachusetts_roads/
      ├── train/sat/  (1108 张 .tiff RGB 1500×1500)
      │        /map/  (1108 张 .tif   灰度 1500×1500)
      ├── val/sat/    (14 张)
      │      /map/
      └── test/sat/   (49 张)
             /map/
    """
    def __init__(self, base_dir, split='train', transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        
        self.sat_dir = os.path.join(base_dir, split, 'sat')
        self.map_dir = os.path.join(base_dir, split, 'map')
        
        if not os.path.exists(self.sat_dir):
            raise FileNotFoundError(f"卫星图像目录不存在: {self.sat_dir}")
        
        # 获取文件列表（sat 为 .tiff, map 为 .tif, 同名）
        sat_files = sorted([f for f in os.listdir(self.sat_dir) if f.endswith('.tiff')])
        self.sample_list = [os.path.splitext(f)[0] for f in sat_files]
        
        print(f"📊 加载 Massachusetts Roads {split} 集: {len(self.sample_list)} 个样本")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        name = self.sample_list[idx]
        
        img_path = os.path.join(self.sat_dir, name + '.tiff')
        mask_path = os.path.join(self.map_dir, name + '.tif')
        
        # 加载卫星图像 (RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载卫星图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载道路掩码 (灰度)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载道路掩码: {mask_path}")
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.uint8)
        
        sample = {
            'image': image,
            'label': mask,
            'case_name': name
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# Roads 的数据增强/测试处理器复用 CVC 的，因为都是 RGB + 二值 mask
Roads_RandomGenerator = CVC_RandomGenerator
Roads_TestGenerator = CVC_TestGenerator


# ============================================================
# ImageCAS Cardiac CCTA 数据集 (3D NIfTI → 2D 切片)
# ============================================================

class CCTA_dataset(Dataset):
    """
    ImageCAS 冠状动脉 CCTA 分割数据集
    
    从 3D NIfTI 体数据中按需提取 2D 轴向切片进行训练。
    数据位于 all_data/，fold 划分由 txt 文件指定。
    
    目录结构:
      cardiac_ccta/
      ├── all_data/{id}.img.nii.gz, {id}.label.nii.gz
      ├── fold_X/train.txt, test.txt (每行一个 case ID)
      └── split_1000.csv
    """
    def __init__(self, data_dir, list_file, transform=None,
                 slices_per_volume=1, max_volumes=None):
        """
        Args:
            data_dir: all_data/ 目录路径
            list_file: train.txt 或 test.txt 路径
            transform: 数据增强变换
            slices_per_volume: 每个 volume 每轮采样的切片数
            max_volumes: 限制加载的 volume 数量（调试用）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.slices_per_volume = slices_per_volume
        
        with open(list_file) as f:
            self.case_ids = [line.strip() for line in f if line.strip()]
        
        if max_volumes and max_volumes < len(self.case_ids):
            self.case_ids = self.case_ids[:max_volumes]
        
        # LRU 缓存
        self._cache = {}
        self._cache_order = []
        self._max_cache = 10
        
        print(f"📊 加载 CCTA 数据集: {len(self.case_ids)} volumes × "
              f"{self.slices_per_volume} slices/vol = {len(self)} samples")
    
    def __len__(self):
        return len(self.case_ids) * self.slices_per_volume
    
    def _load_volume(self, case_id):
        """加载并缓存 3D 体数据"""
        if case_id in self._cache:
            return self._cache[case_id]
        
        import nibabel as nib
        
        img_path = os.path.join(self.data_dir, f'{case_id}.img.nii.gz')
        lbl_path = os.path.join(self.data_dir, f'{case_id}.label.nii.gz')
        
        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.float32)
        
        # CT 窗宽窗位归一化: CCTA 典型窗位 center=200, width=800 → [-200, 600]
        img = np.clip(img, -200, 600)
        img = (img + 200) / 800.0
        
        # 二值化标签
        lbl = (lbl > 0.5).astype(np.float32)
        
        # LRU 缓存管理
        if len(self._cache) >= self._max_cache:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
        self._cache[case_id] = (img, lbl)
        self._cache_order.append(case_id)
        
        return img, lbl
    
    def __getitem__(self, idx):
        vol_idx = idx // self.slices_per_volume
        case_id = self.case_ids[vol_idx]
        
        img, lbl = self._load_volume(case_id)
        
        # 提取随机轴向切片 (NIfTI 通常是 H×W×D 格式)
        z_dim = img.shape[2]
        slice_idx = np.random.randint(0, z_dim)
        
        image = img[:, :, slice_idx]   # (H, W)
        label = lbl[:, :, slice_idx]   # (H, W)
        
        sample = {
            'image': image,
            'label': label,
            'case_name': f'{case_id}_s{slice_idx}'
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class CCTA_TestDataset(Dataset):
    """
    CCTA 测试数据集 - 返回完整 3D 体数据用于体积级评估。
    """
    def __init__(self, data_dir, list_file, max_volumes=None):
        self.data_dir = data_dir
        
        with open(list_file) as f:
            self.case_ids = [line.strip() for line in f if line.strip()]
        
        if max_volumes and max_volumes < len(self.case_ids):
            self.case_ids = self.case_ids[:max_volumes]
        
        print(f"📊 加载 CCTA 测试数据集: {len(self.case_ids)} volumes")
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        import nibabel as nib
        case_id = self.case_ids[idx]
        
        img = nib.load(os.path.join(self.data_dir, f'{case_id}.img.nii.gz')).get_fdata().astype(np.float32)
        lbl = nib.load(os.path.join(self.data_dir, f'{case_id}.label.nii.gz')).get_fdata().astype(np.float32)
        
        img = np.clip(img, -200, 600)
        img = (img + 200) / 800.0
        lbl = (lbl > 0.5).astype(np.float32)
        
        return {'image': img, 'label': lbl, 'case_name': str(case_id)}


class CCTA_RandomGenerator(object):
    """CCTA 训练数据增强 (单通道 CT 切片 → 3 通道)"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 单通道 → 3 通道 (兼容预训练权重)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0).repeat(3, 1, 1)  # (H,W) → (3,H,W)
        
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label, 'case_name': case_name}
        return sample


class CCTA_TestGenerator(object):
    """CCTA 测试数据处理 (单通道 → 3 通道，无增强)"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')
        
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0).repeat(3, 1, 1)
        
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label, 'case_name': case_name}
        return sample
    
