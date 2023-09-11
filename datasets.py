import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import pickle
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_image(path):
    return Image.open(path).convert('RGB')


def load_waveform(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    waveform = audio[:int(samplerate * dur)]

    return waveform, samplerate

def log_mel_spectrogram(waveform, samplerate):
    frequencies, times, spectrogram = signal.spectrogram(waveform, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format in {'vggss', 'vggsound_single'}:
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    elif format == 'vggsound_duet':
        gt_bboxes_raw = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            gt_bboxes_raw[annotation['file']] = annotation['bbox']
        # fns2cls = {item[0]:item[1] for item in csv.reader(open('metadata/vggsound_duet_test.csv'))}
        fns2mix = {item[0]:item[2] for item in csv.reader(open('metadata/vggsound_duet_test.csv'))}
        for annotation in annotations:
            fn = annotation['file']
            fn_mix = fns2mix[fn]
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in gt_bboxes_raw[fn]]
            bboxes_mix = [(np.clip(np.array(bbox_mix), 0, 1) * 224).astype(int) for bbox_mix in gt_bboxes_raw[fn_mix]]
            bboxes_src = [bboxes, bboxes_mix]
            # classes_src = [fns2cls[fn], fns2cls[fn_mix]]
            gt_bboxes[fn] = bboxes_src

    elif format in {'vgginstruments', 'vgginstruments_multi'}:
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt_bboxes[file] = f"{annotation_dir}/{filename}"

    elif format == 'music_solo':
        with open('metadata/music_solo.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(annotation['bbox']), 0, 1) * 224).astype(int)]
            gt_bboxes[annotation['file']] = bboxes

    elif format == 'music_duet':
        with open('metadata/music_duet.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes_src = [annotation['bbox_src1'], annotation['bbox_src2']]
            classes_src = [annotation['class_src1'], annotation['class_src2']]
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in bboxes_src]
            gt_bboxes[annotation['file']] = [bboxes, classes_src]

    return gt_bboxes


def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format in {'vggss', 'music_duet'}:
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map


def mask2gtmap(gt_mask_path):
    with open(gt_mask_path, 'rb') as f:
        gt_mask = pickle.load(f)
    gt_map = cv2.resize(gt_mask, (224,224), interpolation=cv2.INTER_NEAREST)
    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, mode='train', 
            sup_image_path=None, sup_audio_path=None, audio_dur=3., 
            image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr', 
            num_classes=0, class_labels=None, num_mixtures=1, class_labels_ss=None, 
            image_files_ss=None, audio_files_ss=None, all_bboxes_ss=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path

        self.mode = mode
        self.sup_audio_path = sup_audio_path
        self.sup_image_path = sup_image_path

        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format
        self.class_labels = class_labels
        self.num_classes = num_classes

        self.num_mixtures = num_mixtures
        self.class_labels_ss = class_labels_ss
        self.image_files_ss = image_files_ss
        self.audio_files_ss = audio_files_ss
        self.all_bboxes_ss = all_bboxes_ss

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):

        image_path = self.image_path
        audio_path = self.audio_path

        anno = {}

        if self.class_labels is not None:
            class_label = torch.zeros(self.num_classes)
            class_idx = self.class_labels[idx]
            class_label[class_idx] = 1
            anno['class'] = class_label

            if self.class_labels_ss is not None:
                class_label_mix = torch.zeros(self.num_classes)
                class_idx_mix = self.class_labels_ss[idx]
                class_label_mix[class_idx_mix] = 1
                anno['class'] = torch.stack([class_label, class_label_mix])

        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = image_path + self.image_files[idx]
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = audio_path + self.audio_files[idx]
        waveform, samplerate = load_waveform(audio_fn)
        spectrogram = self.audio_transform(log_mel_spectrogram(waveform, samplerate))

        return frame, spectrogram, anno, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path) if fn.endswith('.jpg')}
    if args.trainset in {'music_solo', 'music_duet'}:
        avail_audio_files = []
        for image_file in image_files:
            if image_file[:-10] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    trainset_name_list = args.trainset.split(',')
    trainset = []
    for trainset_name in trainset_name_list:
        if trainset_name in {'vgginstruments_group_0', 'vgginstruments_group_1', 'vgginstruments_group_2', 'vgginstruments_group_3'}:
            trainset_list = [line.split(',')[0] for line in open(f"metadata/avctl_{trainset_name}_train.txt").read().splitlines()]
        else:
            trainset_list = open(f"metadata/{args.trainset}.txt").read().splitlines()
        trainset.extend(trainset_list)
    trainset = set(trainset)
    avail_files = avail_files.intersection(trainset)
    print(f"{len(avail_files)} valid subset files")
    
    avail_files = sorted(list(avail_files))
    audio_files = [dt+'.wav' for dt in avail_files]
    image_files = [dt+'.jpg' for dt in avail_files]
    
    all_bboxes = [[] for _ in range(len(image_files))]

    all_classes = ['flute', 'cello', 'bass_guitar', 'accordion', 'tabla', 'erhu', 'cornet', 'electronic_organ', 'timbales', 
    'acoustic_guitar', 'violin', 'piano', 'banjo', 'glockenspiel', 'steel_guitar', 'vibraphone', 'trumpet', 'zither', 
    'cymbal', 'xylophone', 'harp', 'hammond_organ', 'harpsichord', 'bongo', 'bass_drum', 'mandolin', 'guiro', 
    'saxophone', 'electric_guitar', 'drum_kit', 'clarinet', 'snare_drum', 'ukulele', 'sitar', 'double_bass', 'congas']

    fns2cls = {}
    for trainset_name in trainset_name_list:
        if trainset_name in {'vgginstruments_group_0', 'vgginstruments_group_1', 'vgginstruments_group_2', 'vgginstruments_group_3'}:
            fns2cls.update({line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/avctl_{trainset_name}_train.txt").read().splitlines()})
            
    class_labels = []
    for dt in avail_files:
        cls = all_classes.index(fns2cls[dt])
        class_labels.append(cls)

    num_classes = args.num_class

    print('class_labels:', class_labels[:10])
    print('all_classes:', all_classes)

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='train',
        image_files=image_files,
        audio_files=audio_files,
        all_bboxes=all_bboxes,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        num_classes=num_classes,
        class_labels=class_labels,
        num_mixtures=1
    )


def get_test_dataset(args, mode):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    #  Retrieve list of audio and video files
    testset_name_list = args.testset.split(',')
    testset = []
    for testset_name in testset_name_list:
        if testset_name in {'vgginstruments_group_0', 'vgginstruments_group_1', 'vgginstruments_group_2', 'vgginstruments_group_3'}:
            testset_list = [line.split(',')[0] for line in open(f"metadata/avctl_{testset_name}_{mode}.txt").read().splitlines()]
            testset.extend(testset_list)
    testset = set(testset)
    
    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    if args.testset in {'music_solo', 'music_duet'}:
        avail_audio_files = []
        for image_file in image_files:
            if image_file[:-10] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}

    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)
    print(f"{len(testset)} files for testing")

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    if args.testset in {'music_solo', 'music_duet'}:
        audio_files = [dt[:-10]+'.wav' for dt in testset]
    else:
        audio_files = [dt+'.wav' for dt in testset]
    
    all_classes = ['flute', 'cello', 'bass_guitar', 'accordion', 'tabla', 'erhu', 'cornet', 'electronic_organ', 'timbales', 
    'acoustic_guitar', 'violin', 'piano', 'banjo', 'glockenspiel', 'steel_guitar', 'vibraphone', 'trumpet', 'zither', 
    'cymbal', 'xylophone', 'harp', 'hammond_organ', 'harpsichord', 'bongo', 'bass_drum', 'mandolin', 'guiro', 
    'saxophone', 'electric_guitar', 'drum_kit', 'clarinet', 'snare_drum', 'ukulele', 'sitar', 'double_bass', 'congas']

    fns2cls = {}
    for testset_name in testset_name_list:
        if testset_name in {'vgginstruments_group_0', 'vgginstruments_group_1', 'vgginstruments_group_2', 'vgginstruments_group_3'}:
            fns2cls.update({line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/avctl_{testset_name}_{mode}.txt").read().splitlines()})
            
    class_labels = []
    for dt in testset:
        cls = all_classes.index(fns2cls[dt])
        class_labels.append(cls)

    num_classes = len(list(set(class_labels)))
        
    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='test',
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=5.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        num_classes=num_classes,
        class_labels=class_labels,
        num_mixtures=1
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor

def convert_normalize(tensor, new_mean, new_std):
    raw_mean = IMAGENET_DEFAULT_MEAN
    raw_std = IMAGENET_DEFAULT_STD
    # inverse_normalize with raw mean & raw std
    inverse_mean = [-mean/std for mean, std in zip(raw_mean, raw_std)]
    inverse_std = [1.0/std for std in raw_std]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    # normalize with new mean & new std
    tensor = transforms.Normalize(new_mean, new_std)(tensor)
    return tensor