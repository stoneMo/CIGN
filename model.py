import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.distributed as dist
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from dino.hubconf import *
from grouping import ModalityTrans


class CIGN(nn.Module):
    def __init__(self, tau, dim, dropout_img, dropout_aud, args):
        super(CIGN, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.img_dropout = nn.Dropout(p=dropout_img)

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.audnet.avgpool = nn.Identity()
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)
        # self.aud_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.aud_dropout = nn.Dropout(p=dropout_aud)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

        # hard or soft assignment
        self.unimodal_assgin = args.attn_assign
        unimodal_hard_assignment = True if args.attn_assign == 'hard' else False

        # learnable tokens
        self.num_class = args.num_class
        self.train_stage = args.train_stage
        if self.train_stage == 0:
            self.av_token = nn.Parameter(torch.zeros(args.num_class//1, args.dim))
        elif self.train_stage == 1:
            self.av_token = nn.Parameter(torch.zeros(args.num_class//2, args.dim), requires_grad=False)
            self.av_token_1 = nn.Parameter(torch.zeros(args.num_class//2, args.dim))
        elif self.train_stage == 2:
            self.av_token = nn.Parameter(torch.zeros(args.num_class//3, args.dim), requires_grad=False)
            self.av_token_1 = nn.Parameter(torch.zeros(args.num_class//3, args.dim), requires_grad=False)
            self.av_token_2 = nn.Parameter(torch.zeros(args.num_class//3, args.dim))
        elif self.train_stage == 3:
            self.av_token = nn.Parameter(torch.zeros(args.num_class//4, args.dim))
            self.av_token_1 = nn.Parameter(torch.zeros(args.num_class//4, args.dim))
            self.av_token_2 = nn.Parameter(torch.zeros(args.num_class//4, args.dim))
            self.av_token_3 = nn.Parameter(torch.zeros(args.num_class//4, args.dim))
        
        self.test_stage = args.test_stage

        # uni-modal encoder
        self.audio_encoder = ModalityTrans(
                            args.dim,
                            depth=args.depth_aud,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=args.num_class,
                            num_output_groups=args.num_class,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=True
                        )

        self.visual_encoder = ModalityTrans(
                            args.dim,
                            depth=args.depth_vis,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=args.num_class,
                            num_output_groups=args.num_class,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=False
                        )

        # prediction heads
        self.fc_prob_a = nn.Linear(args.dim, 1)
        self.fc_prob_v = nn.Linear(args.dim, 1)
        self.fc_cls = nn.Linear(args.dim, args.num_class)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
        return loss, Slogits

    def cls_token_loss(self, cls_prob):
        cls_target = torch.arange(0,self.num_class).long().to(cls_prob.device)
        loss = F.cross_entropy(cls_prob, cls_target)
        return loss, cls_target

    def cls_pred_loss(self, prob, target):
        loss = F.binary_cross_entropy(prob, target)
        return loss

    def forward(self, image, audio, mode='train', cls_target=None):
           
        cls_target_v = cls_target.copy()
        cls_target_a = cls_target.copy()

        # Image
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_dropout(img)
        img = self.img_proj(img)
        img = nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio)
        aud = self.aud_dropout(aud)
        aud = self.aud_proj(aud)
        aud = nn.functional.normalize(aud, dim=1)
         
        if self.test_stage == 0:
            av_token = self.av_token
        elif self.test_stage == 1:
            av_token = torch.cat([self.av_token, self.av_token_1], dim=0)
        elif self.test_stage == 2:
            av_token = torch.cat([self.av_token, self.av_token_1, self.av_token_2], dim=0)
        elif self.test_stage == 3:
            av_token = torch.cat([self.av_token, self.av_token_1, self.av_token_2, self.av_token_3], dim=0)

        xv = img.flatten(2,3).permute(0,2,1)
        xv, _, _ = self.visual_encoder(xv, av_token, return_attn=True)
        xa = aud.unsqueeze(1)
        xa, _, _ = self.audio_encoder(xa, av_token, return_attn=True)

        av_cls_prob = self.fc_cls(av_token)                                      
        a_prob = torch.sigmoid(self.fc_prob_a(xa))                                  
        a_pred_prob = a_prob.sum(dim=-1)                                            
        v_prob = torch.sigmoid(self.fc_prob_v(xv))                                  
        v_pred_prob = v_prob.sum(dim=-1)                                            

        cls_token_loss, av_cls_target = self.cls_token_loss(av_cls_prob)
        cls_pred_loss = self.cls_pred_loss(v_pred_prob, cls_target_v) + self.cls_pred_loss(a_pred_prob, cls_target_a)
        
        return a_pred_prob, v_pred_prob, cls_token_loss, cls_pred_loss, av_cls_prob, av_cls_target
        