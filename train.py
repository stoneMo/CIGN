from multiprocessing import reduction
import os
import argparse
import builtins
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

import utils
from model import CIGN
from datasets import get_train_dataset, get_test_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='cign_vggss', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--experiment_name_pre', type=str, default='cign_vggss', help='experiment name of previous stage (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--num_test_samples', default=-1, type=int)
    parser.add_argument('--num_class', default=37, type=int)

    parser.add_argument('--sup_train_data_path', default='', type=str, help='Root directory path of train data w. labels')
    parser.add_argument('--sup_train_gt_path', default='', type=str)
    parser.add_argument('--use_supervised_data', action='store_true')

    # mo-vsl hyper-params
    parser.add_argument('--model', default='cign')
    parser.add_argument('--imgnet_type', default='vitb8')
    parser.add_argument('--audnet_type', default='vitb8')

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    parser.add_argument('--attn_assign', type=str, default='soft', help="type of audio grouping assignment")
    parser.add_argument('--dim', type=int, default=512, help='dimensionality of features')
    parser.add_argument('--depth_aud', type=int, default=3, help='depth of audio transformers')
    parser.add_argument('--depth_vis', type=int, default=3, help='depth of visual transformers')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--lr_schedule", default='cte', help="learning rate schedule")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument("--clip_norm", type=float, default=0, help="gradient clip norm")
    parser.add_argument("--dropout_img", type=float, default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")
    
    parser.add_argument("--train_stage", type=int, default=0, help="training stage")
    parser.add_argument("--test_stage", type=int, default=0, help="testing stage")
    parser.add_argument('--resume_av_token', action='store_true')

    parser.add_argument('--m_img', default=0.9, type=float, metavar='M', help='momentum for imgnet')
    parser.add_argument('--m_aud', default=0.9, type=float, metavar='M', help='momentum for audnet')
    parser.add_argument('--use_momentum', action='store_true')
    parser.add_argument('--use_mom_eval', action='store_true')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # tb writers
    tb_writer = SummaryWriter(model_dir)

    # logger
    log_fn = f"{model_dir}/train.log"
    def print_and_log(*content, **kwargs):
        # suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(log_fn, 'a') as f:
            f.write(msg+'\n')
    builtins.print = print_and_log

    # Create model
    if args.model.lower() == 'CIGN':
        model = CIGN(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, args)
    else:
        raise ValueError

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    optimizer, _ = utils.build_optimizer_and_scheduler_adam(model, args)

    # load av tokens trained from previous stage
    if args.resume_av_token:
        model_dir_pre = os.path.join(args.model_dir, args.experiment_name_pre)
        ckp = torch.load(os.path.join(model_dir_pre, 'best.pth'), map_location='cpu')
        pretrained_tokens = {k: v for k, v in ckp['model'].items() if k.startswith('module.av_token')}
        print('pretrained_tokens:', pretrained_tokens.keys())
        model_dict = model.state_dict()
        model_dict.update(pretrained_tokens)
        model.load_state_dict(model_dict, strict=False)

    # Resume current stage if possible
    start_epoch, best_aacc, best_vacc, best_avacc = 0, 0., 0., 0.
    if os.path.exists(os.path.join(model_dir, 'best.pth')):
        ckp = torch.load(os.path.join(model_dir, 'best.pth'), map_location='cpu')
        start_epoch, best_aacc, best_vacc, best_avacc = ckp['epoch'], ckp['best_AACC'], ckp['best_VACC'], ckp['best_AVACC']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')

    # Dataloaders
    traindataset = get_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    valdataset = get_test_dataset(args, mode='val')
    val_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)

    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    for stage_idx in range(args.test_stage+1):
        a_acc, v_acc, av_acc = validate(val_loader, model, args, test_stage=stage_idx)
        print(f'Test Stage (epoch {start_epoch}): {stage_idx}')
        print(f'Audio Acc (epoch {start_epoch}): {a_acc}')
        print(f'Visual Acc (epoch {start_epoch}): {v_acc}')
        print(f'Audio-visual Acc (epoch {start_epoch}): {av_acc}')

    print(f'best_AACC: {best_aacc}')
    print(f'best_VACC: {best_vacc}')
    print(f'best_AVACC: {best_avacc}')

    metric_list = [[] for _ in range(3)]

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args, tb_writer)

        # Evaluate
        a_acc_list, v_acc_list, av_acc_list = [], [], []
        for stage_idx in range(args.test_stage+1):
            a_acc, v_acc, av_acc = validate(val_loader, model, args, test_stage=stage_idx)
            print(f'Test Stage (epoch {epoch+1}): {stage_idx}')
            print(f'Audio Acc (epoch {epoch+1}): {a_acc}')
            print(f'Visual Acc (epoch {epoch+1}): {v_acc}')
            print(f'Audio-visual Acc (epoch {epoch+1}): {av_acc}')

            tb_writer.add_scalar(f'Stage {stage_idx} Audio Acc', a_acc, epoch)
            tb_writer.add_scalar(f'Stage {stage_idx} Visual Acc', v_acc, epoch)
            tb_writer.add_scalar(f'Stage {stage_idx} Audio-visual Acc', av_acc, epoch)

            a_acc_list.append(a_acc)
            v_acc_list.append(v_acc)
            av_acc_list.append(av_acc)

        a_acc, v_acc, av_acc = np.mean(a_acc_list), np.mean(v_acc_list), np.mean(av_acc_list)

        if av_acc >= best_avacc:
           best_aacc, best_vacc, best_avacc = a_acc, v_acc, av_acc
        
        print(f'best_AACC: {best_aacc}')
        print(f'best_VACC: {best_vacc}')
        print(f'best_AVACC: {best_avacc}')

        metric_list[0].append(a_acc)
        metric_list[1].append(v_acc)
        metric_list[2].append(av_acc)

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_AACC': best_aacc,
                   'best_VACC': best_vacc,
                   'best_AVACC': best_avacc}
            torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
            if av_acc == best_avacc:
                torch.save(ckp, os.path.join(model_dir, 'best.pth'))
            print(f"Model saved to {model_dir}")

    np.save(os.path.join(model_dir, 'metrics.npy'), np.array(metric_list))


    # ============Test============

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    ckp = torch.load(ckp_fn, map_location='cpu')
    model.load_state_dict(ckp['model'])
    print(f'loaded from {ckp_fn}')

    testdataset = get_test_dataset(args, mode='test')
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)

    a_acc_list = []
    v_acc_list = []
    av_acc_list = []
    
    for stage_idx in range(args.test_stage+1):
        a_acc, v_acc, av_acc = validate(test_loader, model, args, test_stage=stage_idx)
        print(f'Test Stage (epoch {epoch+1}): {stage_idx}')
        print(f'Audio Acc (epoch {epoch+1}): {a_acc}')
        print(f'Visual Acc (epoch {epoch+1}): {v_acc}')
        print(f'Audio-visual Acc (epoch {epoch+1}): {av_acc}')

        tb_writer.add_scalar(f'Stage {stage_idx} Audio Acc', a_acc, epoch)
        tb_writer.add_scalar(f'Stage {stage_idx} Visual Acc', v_acc, epoch)
        tb_writer.add_scalar(f'Stage {stage_idx} Audio-visual Acc', av_acc, epoch)

        a_acc_list.append(a_acc)
        v_acc_list.append(v_acc)
        av_acc_list.append(av_acc)

    a_acc, v_acc, av_acc = np.mean(a_acc_list), np.mean(v_acc_list), np.mean(av_acc_list)

    print(f'Average_AACC: {a_acc}')
    print(f'Average_VACC: {v_acc}')
    print(f'Average_AVACC: {av_acc}')


def cal_precision_recall_token(target, output):
    precison = precision_score(y_true=target, y_pred=output, average='weighted')
    recall = recall_score(y_true=target, y_pred=output, average='weighted')
    f1 = f1_score(y_true=target, y_pred=output, average='weighted')
    return precison, recall, f1


def train(train_loader, model, optimizer, epoch, args, writer):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')
    loss_loc_mtr = AverageMeter('Loc Loss', ':.3f')
    loss_token_mtr = AverageMeter('Token Loss', ':.3f')
    loss_pred_mtr = AverageMeter('Pred Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr, loss_loc_mtr, loss_token_mtr, loss_pred_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, spec, anno, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step = i + len(train_loader) * epoch
        utils.adjust_learning_rate(optimizer, epoch + i / len(train_loader), args)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            label = anno['class'].cuda(args.gpu, non_blocking=True)

        _, _, cls_token_loss, cls_pred_loss, av_cls_prob, av_cls_target = model(image.float(), spec.float(), cls_target=label, mode='train')
        loss = cls_token_loss + cls_pred_loss

        loss_mtr.update(loss.item(), image.shape[0])
        loss_token_mtr.update(cls_token_loss.item(), image.shape[0])
        loss_pred_mtr.update(cls_pred_loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        if args.clip_norm != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # clip gradient

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('loss', loss_mtr.avg, global_step)
        writer.add_scalar('Token loss', loss_token_mtr.avg, global_step)
        writer.add_scalar('Pred loss', loss_pred_mtr.avg, global_step)

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        del loss


def validate(test_loader, model, args, test_stage=0):
    model.train(False)

    a_acc_list = []
    v_acc_list = []
    av_acc_list = []

    for step, (image, spec, bboxes, name) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            label = bboxes['class'].cuda(args.gpu, non_blocking=True)

        a_pred_prob, v_pred_prob = model(image.float(), spec.float(), cls_target=label, mode='test')[:2]
        
        a_pred = a_pred_prob.argmax(dim=1).detach().cpu().numpy()
        v_pred = v_pred_prob.argmax(dim=1).detach().cpu().numpy()
        av_pred = (a_pred_prob * v_pred_prob).argmax(dim=1).detach().cpu().numpy()

        target = label.argmax(dim=1).detach().cpu().numpy()

        eval_index = (target < ((test_stage+1) * 9)) * (target >= ((test_stage) * 9))

        # calculate
        if eval_index.sum() != 0:
            a_acc = ((a_pred == target)*eval_index).sum() / eval_index.sum()
            v_acc = ((v_pred == target)*eval_index).sum() / eval_index.sum()
            av_acc = ((av_pred == target)*eval_index).sum() / eval_index.sum()

            a_acc_list.append(a_acc)
            v_acc_list.append(v_acc)
            av_acc_list.append(av_acc)
    
    a_acc, v_acc, av_acc = np.mean(a_acc_list), np.mean(v_acc_list), np.mean(av_acc_list)
    
    return a_acc, v_acc, av_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
