# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import log
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


@register_criterion("cross_modal_criterion_with_ctr_revise")
class CrossModalCriterionWithCTRRevise(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            report_modal_similarity=False,
            contrastive_weight=0.0,
            contrastive_temperature=1.0,
            use_dual_ctr=False,
            use_v2t_ctr=False,
            ctr_dropout_rate=0.0
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.report_modal_similarity = report_modal_similarity

        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        self.use_dual_ctr = use_dual_ctr
        self.use_v2t_ctr = use_v2t_ctr
        self.ctr_dropout_rate = ctr_dropout_rate
        self.ctr_strategy = task.args.contrastive_strategy
        self.ctr_align = task.args.contrastive_align

        # if self.ctr_strategy == "mean+mlp" or "cls+mlp":
        #     self.proj_dim = 128
        #     self.feature_dim = task.args.encoder_embed_dim
        #
        #     self.video_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
        #                                           nn.BatchNorm1d(self.feature_dim),
        #                                           nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
        #                                           BatchNorm1dNoBias(self.proj_dim))
        #     self.text_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
        #                                          nn.BatchNorm1d(self.feature_dim),
        #                                          nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
        #                                          BatchNorm1dNoBias(self.proj_dim))

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--report-modal-similarity', action='store_true',
                            help='report accuracy metric')

        parser.add_argument('--contrastive-weight', default=0., type=float,
                            help='the weight of contrastive loss')
        parser.add_argument('--contrastive-temperature', default=1.0, type=float,
                            help='the temperature in the contrastive loss')
        parser.add_argument("--use-dual-ctr", action="store_true",
                            help="if we want to use dual contrastive loss")
        parser.add_argument("--use-v2t-ctr", action="store_true",
                            help="if we want to use video find text")
        parser.add_argument("--ctr-dropout-rate", default=0., type=float,
                            help='the dropout rate of hidden units')
        parser.add_argument("--contrastive-strategy", default="mean", type=str, )
        parser.add_argument("--contrastive-align", default="bottom", type=str, )

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        if model.training:
            contrastive_loss, text_hidden, video_hidden = self.compute_contrastive_loss(net_output,
                                                                                        reduce=reduce)
        else:
            contrastive_loss, text_hidden, video_hidden = torch.tensor(0.0), None, None

        label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if label_smoothed_nll_loss is not None:
            loss = label_smoothed_nll_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = contrastive_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "contrastive_loss": contrastive_loss.data,
            "sample_size": sample_size,
            "gpu_nums": 1,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.report_modal_similarity:
            if text_hidden is not None:
                text_h = text_hidden.detach()
                video_h = video_hidden.detach()

                sim = torch.cosine_similarity(text_h, video_h, dim=-1)

                logging_output["modal_similarity"] = utils.item(sim.mean().data)

            else:
                logging_output["modal_similarity"] = -1

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def compute_contrastive_loss(self, net_output,
                                 reduce=True):
        out, extra = net_output

        text_hidden = extra['bottom_text_proj_h']  # B, t_len , C
        video_hidden = extra['bottom_video_proj_h']

        if self.ctr_strategy == "mean+mlp":

            batch_size, hidden_size = text_hidden.size()
            logits = F.cosine_similarity(text_hidden.expand((batch_size, batch_size, hidden_size)),
                                         video_hidden.expand((batch_size, batch_size, hidden_size)).transpose(0, 1),
                                         dim=-1)

            logits /= self.contrastive_temperature

            if self.use_dual_ctr:
                loss_text = -torch.nn.LogSoftmax(0)(logits).diag()
                loss_video = -torch.nn.LogSoftmax(1)(logits).diag()
                loss = loss_text + loss_video
            elif self.use_v2t_ctr:
                loss = -torch.nn.LogSoftmax(1)(logits).diag()
            else:
                loss = -torch.nn.LogSoftmax(0)(logits).diag()
            if reduce:
                loss = loss.sum()

        return loss, text_hidden, video_hidden

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:

        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)

        GPU_nums = sum(log.get('gpu_nums', 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "contrasitve_loss", contrastive_loss_sum / nsentences / math.log(2), nsentences, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        modal_similarity_sum = sum(log.get("modal_similarity", 0) for log in logging_outputs)

        metrics.log_scalar(
            "modal_similarity", modal_similarity_sum / len(logging_outputs) / GPU_nums, round=5
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
