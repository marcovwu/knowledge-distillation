
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLoss(nn.Module):
    def __init__(self, reduction='sum', Aobj=4, Abg=16, const=0.025):
        super(FeatureLoss, self).__init__()
        self.Aobj = Aobj
        self.Abg = Abg
        self.const = const
        self.reduction = reduction

    def _get_mask_from_ground_truth(self, teacher, gt_instances, mode=0):
        gt_masks = [torch.zeros(list(spl.size())) for spl in teacher]
        for i in range(len(gt_instances)):
            boxes = gt_instances[i][:, 2:]  # N x 4
            # gt_classes = gt_instances[i][:, 1]  # N in [0, self.num_classes]
            for gt_mask in gt_masks:
                for box in boxes:
                    x1 = self.find_inds(box[0] * gt_mask.size()[2], mode=mode)
                    y1 = self.find_inds(box[1] * gt_mask.size()[3], mode=mode)
                    x2 = self.find_inds(box[2] * gt_mask.size()[2], mode=mode)
                    y2 = self.find_inds(box[3] * gt_mask.size()[3], mode=mode)

                    gt_mask[i, :, x1:x2, y1:y2] = torch.ones_like(gt_mask[i, :, x1:x2, y1:y2])

        return gt_masks

    def find_inds(self, index, mode=0):
        if mode == 0:
            return int(index)
        elif mode == 1:
            return int(index) + 1 if index % 1 >= 0.5 else int(index)
        else:
            return int(index)

    def forward(self, gt_instances, student, teacher, mode=0):
        loss_pos = 0
        loss_neg = 0
        # no = len(teacher)
        # balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        masks = self._get_mask_from_ground_truth(teacher, gt_instances, mode=mode)
        neg_masks = [torch.ones_like(mk) - mk for mk in masks]
        for mi, (mask, neg_mask, stu, tea) in enumerate(zip(masks, neg_masks, student, teacher)):
            for si in range(tea.size()[0]):
                if mask[si].sum() != 0:
                    loss_pos = loss_pos + (mask[si].cuda() * (stu[si] - tea[si]) ** 2).sum() / (
                        mask[si].cuda()).sum()  # * balance[si]
                if neg_mask[si].sum() != 0:
                    loss_neg = loss_neg + (neg_mask[si].cuda() * (stu[si] - tea[si]) ** 2).sum() / (
                        neg_mask[si].cuda()).sum()  # * balance[si]
        loss_pos = self.Aobj * loss_pos / 2
        loss_neg = self.Abg * loss_neg / 2
        fea_loss = (loss_pos + loss_neg) * self.const / len(masks)
        return fea_loss, (fea_loss / tea.size()[0]).detach().unsqueeze(0)


class AttentionLoss(nn.Module):
    def __init__(self, ATT_PW=1.0, A=4 * 10 ** -4, B=2 * 10 ** -2, Hinton_T=0.5, ATT=1):
        super(AttentionLoss, self).__init__()
        self.att_pw = ATT_PW
        self.T = Hinton_T
        self.A = A
        self.B = B
        self.att = ATT

    def forward(self, targets, student, teacher):
        lam = 0
        lat = 0
        # BCEatt = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.att_pw])).to(targets.device)
        MSEloss = nn.MSELoss(size_average=None, reduce=None, reduction='sum')
        for stu, tea in zip(student, teacher):
            slam = 0
            BS = stu.size()[0]
            C = stu.size()[1]
            HW = stu.size()[2] * stu.size()[3]
            Gs_stu = stu.abs().sum(1) / C
            Gc_stu = stu.abs().sum(dim=(2, 3)) / HW
            Gs_tea = tea.abs().sum(1) / C
            Gc_tea = tea.abs().sum(dim=(2, 3)) / HW

            # lat += BCEatt(Gs_stu, Gs_tea) + BCEatt(Gc_stu, Gc_tea)  # BCE
            # lat += (((Gs_stu - Gs_tea)**2).sum() + ((Gc_stu - Gc_tea)**2).sum())
            lat += MSEloss(Gs_stu, Gs_tea) + MSEloss(Gc_stu, Gc_tea)  # MSE

            Ms = torch.ones_like(Gs_stu)
            for bs in range(BS):
                Tx = ((Gs_stu[bs] + Gs_tea[bs]) / self.T)
                Ms[bs] = HW * (Tx - Tx.max()).exp() / ((Tx - Tx.max()).exp()).sum()
            # Ms = HW * ((Gs_stu + Gs_tea) / self.T).softmax(dim=1)
            Mc = C * ((Gc_stu + Gc_tea) / self.T).softmax(dim=1)

            for c in range(C):
                slam += (Mc[:, c] * ((stu[:, c] - tea[:, c]) ** 2 * Ms).sum(dim=(1, 2))).sum()
            lam += slam ** (1 / 2)
        att_loss = self.att * (self.A * lat + self.B * lam) / BS / len(student)

        return att_loss * BS, att_loss.detach().unsqueeze(0)


class NonLocalLoss(nn.Module):
    def __init__(self, R=4 * 10 ** -4, NON=0.05):
        super(NonLocalLoss, self).__init__()
        self.r = R
        self.non = NON

    def forward(self, model, student, teacher):
        lnon = 0
        MSEloss = nn.MSELoss(size_average=None, reduce=None, reduction='sum')
        for i, (stu, tea) in enumerate(zip(student, teacher)):

            bs, c, w, h = stu.shape  # bs, c, w, h
            stu_s = model.model[-1].si[i](stu)
            stu_f = model.model[-1].fi[i](stu)
            stu_g = model.model[-1].gi[i](stu)
            F_stu = stu_s.view(bs, c, w * h).permute(0, 2, 1).contiguous().matmul(stu_f.view(bs, c, w * h))
            # .softmax(dim=1)

            NF_stu = torch.ones_like(F_stu)
            for bs_ in range(bs):
                if (F_stu[bs_] - F_stu[bs_].max()).exp().sum() != 0:
                    NF_stu[bs_] = (F_stu[bs_] - F_stu[bs_].max()).exp() / (F_stu[bs_] - F_stu[bs_].max()).exp().sum()
            G_stu = stu_g.view(bs, c, w * h)
            FSG_stu = model.model[-1].fsg[i](G_stu.matmul(NF_stu).view(bs, c, w, h))
            out_stu = (stu + FSG_stu).sum(dim=(2, 3)) / (h * w)  # bs, c

            bs, c, w, h = tea.shape  # bs, c, w, h
            tea_s = model.model[-1].si[i](tea)
            tea_f = model.model[-1].fi[i](tea)
            tea_g = model.model[-1].gi[i](tea)
            F_tea = tea_s.view(bs, c, w * h).permute(0, 2, 1).contiguous().matmul(tea_f.view(bs, c, w * h))
            # .softmax(dim=1)

            NF_tea = torch.ones_like(F_tea)
            for bs_ in range(bs):
                if (F_tea[bs_] - F_tea[bs_].max()).exp().sum() != 0:
                    NF_tea[bs_] = (F_tea[bs_] - F_tea[bs_].max()).exp() / (F_tea[bs_] - F_tea[bs_].max()).exp().sum()
            G_tea = tea_g.view(bs, c, w * h)
            FSG_tea = model.model[-1].fsg[i](G_tea.matmul(NF_tea).view(bs, c, w, h))
            out_tea = (tea + FSG_tea).sum(dim=(2, 3)) / (h * w)  # bs, c

            lnon += MSEloss(out_stu, out_tea)

        non_loss = self.non * self.r * lnon / bs / len(student)

        return non_loss * bs, non_loss.detach().unsqueeze(0)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def Save_Checkpoint(state, last, last_path, best, best_path, is_best):
    if os.path.exists(last):
        shutil.rmtree(last)
    last_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(last_path, 'ckpt.pth'))

    if is_best:
        if os.path.exists(best):
            shutil.rmtree(best)
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(state, os.path.join(best_path, 'ckpt.pth'))


class DynamicCenterNormLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, nd_loss_factor=1.0):
        super(DynamicCenterNormLoss, self).__init__()
        self.num_classes = num_classes
        self.nd_loss_factor = nd_loss_factor
        self.class_centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, s_emb, t_emb, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)

        loss = 0.0
        for s, t, label in zip(s_emb, t_emb, labels):
            center = self.class_centers[label]
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm

        nd_loss = loss * self.nd_loss_factor

        return nd_loss / len(labels)


class DirectNormLoss(nn.Module):

    def __init__(self, num_class=1000, nd_loss_factor=1.0):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        self.nd_loss_factor = nd_loss_factor

    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss

    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_loss_factor

        return nd_loss / len(labels)


class KDLoss(nn.Module):
    """ Distilling the Knowledge in a Neural Network https://arxiv.org/pdf/1503.02531.pdf """

    def __init__(self, kl_loss_factor=1.0, T=1.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1),
                           F.softmax(t_out / self.T, dim=1),
                           # reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha=1.0, beta=1.0, T=1.0):
        super(DKDLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.T = T

    def dkd_loss(self, s_logits, t_logits, labels):
        gt_mask = self.get_gt_mask(s_logits, labels)
        other_mask = self.get_other_mask(s_logits, labels)
        s_pred = F.softmax(s_logits / self.T, dim=1)
        t_pred = F.softmax(t_logits / self.T, dim=1)
        s_pred = self.cat_mask(s_pred, gt_mask, other_mask)
        t_pred = self.cat_mask(t_pred, gt_mask, other_mask)
        s_log_pred = torch.log(s_pred)
        tckd_loss = (
            F.kl_div(s_log_pred, t_pred, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            t_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            s_logits / self.T - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.T**2)
            / labels.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def get_gt_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1).bool()
        return mask

    def get_other_mask(self, logits, labels):
        labels = labels.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def forward(self, s_logits, t_logits, labels):
        loss = self.dkd_loss(s_logits, t_logits, labels)

        return loss


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, ori_student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = ori_student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def SKLD(stu, tea, temperature=1.0):
    stu = F.softmax(stu / temperature, dim=1)
    tea = F.softmax(tea / temperature, dim=1)
    return F.kl_div(stu.log(), tea)  # , reduction='batchmean')


def DIFFKLD(stu, tea, temperature=1.0):
    # [diff]
    stu = (stu.unsqueeze(1) - stu.unsqueeze(0)).view(-1, stu.shape[1])
    tea = (tea.unsqueeze(1) - tea.unsqueeze(0)).view(-1, tea.shape[1])
    return F.kl_div(F.log_softmax(stu), F.softmax(tea), reduction='batchmean')
