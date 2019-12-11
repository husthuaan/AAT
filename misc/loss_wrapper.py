import torch
import numpy as np

import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method':'sample'}, mode='sample')

            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()

        if self.opt.caption_model == 'aat':
            all_aat_loss = torch.stack(self.model.all_att_cost).t()
            if not sc_flag:
                mask_ = masks[:,:all_aat_loss.size()[1]]
            else:
                mask_ = (torch.cat((gen_result.new_ones(gen_result.size(0),1), gen_result), dim=1)>0)[:,:all_aat_loss.size()[1]]
            aat_loss = (all_aat_loss * mask_.float()).sum(1).mean()
            out['aat_loss'] = aat_loss
            out['att_step'] = self.model.all_att_step
            out['avg_att_time'] = (np.array(self.model.all_att_step).transpose() * mask_.cpu().numpy()).sum()/mask_.cpu().numpy().sum()
            out['loss_'] = loss.clone()
            loss += self.opt.aat_lambda * aat_loss

        out['loss'] = loss
        return out
