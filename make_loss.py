import torch.nn.functional as F

from .triplet_loss import EU_MSELoss
from .kd_loss import DynamicCenterNormLoss, KDLoss, DKDLoss, DINOLoss, DIFFKLD, SKLD  # , DirectNormLoss
from .crcd.criterion import CRCDLoss
from .nkd_loss import NKDLoss
from .uskd_loss import USKDLoss
from .reviewkd import hcl


def compute_kd_loss(cfg, kd_losses, feat, target, epoch):
    KD_LOSS = 0
    for inx, (kd_type, kd_loss) in enumerate(kd_losses.items()):
        # [feature]
        if kd_type == 'dino':
            loss = kd_loss(feat['stu'], feat['tea'], epoch)  # softmax
        elif kd_type == 'mse' or kd_type == 'bce' or kd_type == 'kld' or kd_type == 'sml':
            loss = kd_loss(feat['stu'], feat['tea'])
        elif kd_type == 'dcn':
            loss = kd_loss(feat['stu'], feat['tea'], target.detach().clone())
        elif kd_type == 'crcd':
            loss = kd_loss(feat['stu'], feat['tea'], target)
        elif kd_type == 'rkd':
            loss = kd_loss(feat['feats_stu'], feat['feats_tea'][1:])

        # [logtics]
        elif kd_type == 'diff' or kd_type == 'kld' or kd_type == 'skld' or kd_type == 'eu_mse':
            loss = kd_loss(feat['score_stu'], feat['score_tea'])
        elif kd_type == 'dkd' or kd_type == 'nkd':
            loss = kd_loss(feat['score_stu'], feat['score_tea'], target.detach().clone())
        elif kd_type == 'uskd':
            loss = kd_loss(feat['fea_stu'], feat['score_stu'], target.detach().clone())
        elif kd_type == 'uskd_tea':
            loss = kd_loss(feat['fea_stu'], feat['score_stu'], feat['score_tea'].max(1)[1])
        else:
            print('%s loss not found!!' % kd_type)
        KD_LOSS += loss * cfg.MODEL.KD_LOSS_WEIGHT[inx]
    return KD_LOSS


def build_kd_losses(cfg, num_classes, num_data, device="cuda"):
    # cfg
    class Opt:
        embed_type, n_data = 'linear', num_data
        s_dim, t_dim, feat_dim = cfg.MODEL.FEAT_DIM, cfg.MODEL.FEAT_DIM, cfg.MODEL.FEAT_DIM
        nce_k, nce_t, nce_m = 500, 0.05, 0.5

    # mapping
    # dkd_loss = DKDLoss(alpha=args.dkd_alpha, beta=args.dkd_beta, T=args.t).to(device)
    # nd_loss = DirectNormLoss(num_class=num_class, nd_loss_factor=args.nd_loss_factor).to(device)
    # KD loss
    # div_loss = kd_loss(s_logits, t_logits, labels) * min(1.0, epoch/args.warm_up)
    # ND loss
    # norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * min(1.0, epoch/args.warm_up)
    kd_loss_mapping = {
        # dino: total number of crops = 2 global crops + local_crops_number
        'dino': lambda: DINOLoss(cfg.MODEL.FEAT_DIM, 2, 0.04, 0.04, 30, cfg.SOLVER.MAX_EPOCHS).to(device),
        'bce': lambda: F.binary_cross_entropy_with_logits,
        'mse': lambda: F.mse_loss,
        'sml': lambda: F.smooth_l1_loss,
        # tckd = kld(student[sum(target), sum(non-target)], teacher[sum(target), sum(non-target)])
        # nckd = kld(student[logtic(target) - 1000], teacher[logtic(target) - 1000])
        'dkd': lambda: DKDLoss(alpha=1.0, beta=1.0, T=1.0).to(device),
        'kld': lambda: KDLoss(kl_loss_factor=1.0, T=1.0),
        'diff': lambda: DIFFKLD,
        'skld': lambda: SKLD,
        'eu_mse': lambda: EU_MSELoss,
        'dcn': lambda: DynamicCenterNormLoss(num_classes=num_classes, feat_dim=cfg.MODEL.FEAT_DIM, nd_loss_factor=1.0
                                             ).to(device),
        'crcd': lambda: CRCDLoss(Opt()).to(device),
        'nkd': lambda: NKDLoss(),
        # kd_losses[distill_type] = USKDLoss(channel=cfg.MODEL.FEAT_DIM, num_classes=num_classes).to(device)  # 160
        'uskd': lambda: USKDLoss(channel=cfg.MODEL.CLS_FEA_NUM, num_classes=num_classes).to(device),  # 320
        'rkd': lambda: hcl,
    }

    kd_losses = {
        distill_type: kd_loss_mapping[distill_type]() for distill_type in cfg.MODEL.METRIC_LOSS_TYPE.split('-')[1:]}
    return kd_losses


def make_loss(cfg, num_classes, num_data):    # modified by gu
    # Knowledge Distillation Loss
    kd_losses = build_kd_losses(cfg, num_classes, num_data)

    def loss_func(score, feat, fea, target, epoch, kd_losses=kd_losses):
        # knowledge distillation loss
        KD_LOSS = compute_kd_loss(cfg, kd_losses, feat, target, epoch)
        return KD_LOSS

    return loss_func
