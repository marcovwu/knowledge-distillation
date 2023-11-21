import torch
import torch.nn as nn


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
