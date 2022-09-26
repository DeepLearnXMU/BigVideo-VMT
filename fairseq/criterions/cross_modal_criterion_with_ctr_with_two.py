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


@register_criterion("cross_modal_criterion_with_ctr_with_two")
class CrossModalCriterionWithCTRWithTwo(FairseqCriterion):
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

        if self.ctr_strategy == "mean+mlp" or "cls+mlp":
            self.proj_dim = 128
            self.feature_dim = task.args.encoder_embed_dim

            self.src_video_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
                                                      nn.BatchNorm1d(self.feature_dim),
                                                      nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
                                                      BatchNorm1dNoBias(self.proj_dim))
            self.src_text_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
                                                     nn.BatchNorm1d(self.feature_dim),
                                                     nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
                                                     BatchNorm1dNoBias(self.proj_dim))
            self.tgt_video_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
                                                      nn.BatchNorm1d(self.feature_dim),
                                                      nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
                                                      BatchNorm1dNoBias(self.proj_dim))
            self.tgt_text_projection = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim, bias=False),
                                                     nn.BatchNorm1d(self.feature_dim),
                                                     nn.ReLU(), nn.Linear(self.feature_dim, self.proj_dim, bias=False),
                                                     BatchNorm1dNoBias(self.proj_dim))

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
            src_contrastive_loss, tgt_contrastive_loss, src_text_hidden, src_video_hidden, tgt_text_hidden, tgt_video_hidden = self.compute_contrastive_loss(
                net_output,
                reduce=reduce)
        else:
            src_contrastive_loss, tgt_contrastive_loss, src_text_hidden, src_video_hidden, tgt_text_hidden, tgt_video_hidden = torch.tensor(
                0.0), torch.tensor(0.0), None, None, None, None

        label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        if label_smoothed_nll_loss is not None:
            loss = label_smoothed_nll_loss + self.contrastive_weight * src_contrastive_loss + self.contrastive_weight * tgt_contrastive_loss
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
            "src_contrastive_loss": src_contrastive_loss.data,
            "tgt_contrastive_loss": src_contrastive_loss.data,
            "sample_size": sample_size,
            "gpu_nums": 1,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.report_modal_similarity:
            if src_text_hidden is not None:
                src_text_h = src_text_hidden.detach()
                src_video_h = src_video_hidden.detach()

                sim = torch.cosine_similarity(src_text_h, src_video_h, dim=-1)

                logging_output["src2video_modal_similarity"] = utils.item(sim.mean().data)

            else:
                logging_output["src2video_modal_similarity"] = -1

            if tgt_text_hidden is not None:
                src_text_h = src_text_hidden.detach()
                tgt_text_h = tgt_text_hidden.detach()
                tgt_video_h = tgt_video_hidden.detach()
                sim1 = torch.cosine_similarity(tgt_text_h, tgt_video_h, dim=-1)
                logging_output["tgt2video_modal_similarity"] = utils.item(sim1.mean().data)

                sim2 = torch.cosine_similarity(tgt_text_h, src_text_h, dim=-1)
                logging_output["src2tgt_modal_similarity"] = utils.item(sim2.mean().data)


            else:
                logging_output["tgt2video_modal_similarity"] = -1

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

        bottom_text_h = extra['bottom_text_h']  # B, t_len , C
        bottom_video_h = extra['bottom_video_h']
        top_text_h = extra['top_text_h']  # B, t_len , C
        top_video_h = extra['top_video_h']
        tgt_bottom_text_h = extra['tgt_bottom_text_h']
        tgt_top_text_h = extra['tgt_top_text_h']

        text_padding_mask = extra["text_padding_mask"]  # B, t_len
        video_padding_mask = extra["video_padding_mask"]
        tgt_text_padding_mask = extra["tgt_text_padding_mask"]

        text_padding_mask = (~text_padding_mask).float()
        video_padding_mask = (~video_padding_mask).float()
        tgt_text_padding_mask = (~tgt_text_padding_mask).float()

        if self.ctr_strategy == "mean":

            if self.ctr_align == "bottom2bottom":
                src_text_hidden = (bottom_text_h * text_padding_mask.unsqueeze(-1)).sum(dim=1) / text_padding_mask.sum(
                    dim=1).unsqueeze(
                    -1)  # Bx H

                src_video_hidden = (bottom_video_h * video_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

                tgt_text_hidden = (tgt_bottom_text_h * tgt_text_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / tgt_text_padding_mask.sum(
                    dim=1).unsqueeze(-1)
                tgt_video_hidden = (bottom_video_h * video_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

            elif self.ctr_align == "top2top":
                src_text_hidden = (top_text_h * text_padding_mask.unsqueeze(-1)).sum(dim=1) / text_padding_mask.sum(
                    dim=1).unsqueeze(
                    -1)  # Bx H

                src_video_hidden = (top_video_h * video_padding_mask.unsqueeze(-1)).sum(dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

                tgt_text_hidden = (tgt_top_text_h * tgt_text_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / tgt_text_padding_mask.sum(
                    dim=1).unsqueeze(-1)
                tgt_video_hidden = (top_video_h * video_padding_mask.unsqueeze(-1)).sum(dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)


            src_batch_size, src_hidden_size = src_text_hidden.size()
            src_logits = F.cosine_similarity(src_text_hidden.expand((src_batch_size, src_batch_size, src_hidden_size)),
                                             src_video_hidden.expand(
                                                 (src_batch_size, src_batch_size, src_hidden_size)).transpose(0, 1),
                                             dim=-1)
            src_logits /= self.contrastive_temperature
            src_loss = -torch.nn.LogSoftmax(0)(src_logits).diag()


            tgt_batch_size, tgt_hidden_size = tgt_text_hidden.size()

            tgt_logits = F.cosine_similarity(tgt_text_hidden.expand((tgt_batch_size, tgt_batch_size, tgt_hidden_size)),
                                             tgt_video_hidden.expand(
                                                 (tgt_batch_size, tgt_batch_size, tgt_hidden_size)).transpose(0, 1),
                                             dim=-1)
            tgt_logits /= self.contrastive_temperature
            tgt_loss = -torch.nn.LogSoftmax(0)(tgt_logits).diag()

            if reduce:
                src_loss = src_loss.sum()
                tgt_loss = tgt_loss.sum()

        elif self.ctr_strategy == "mean+mlp":

            if self.ctr_align == "bottom2bottom":
                src_text_hidden = (bottom_text_h * text_padding_mask.unsqueeze(-1)).sum(dim=1) / text_padding_mask.sum(
                    dim=1).unsqueeze(
                    -1)  # Bx H

                src_video_hidden = (bottom_video_h * video_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

                tgt_text_hidden = (tgt_bottom_text_h * tgt_text_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / tgt_text_padding_mask.sum(
                    dim=1).unsqueeze(-1)
                tgt_video_hidden = (bottom_video_h * video_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

            elif self.ctr_align == "top2top":
                src_text_hidden = (top_text_h * text_padding_mask.unsqueeze(-1)).sum(dim=1) / text_padding_mask.sum(
                    dim=1).unsqueeze(
                    -1)  # Bx H

                src_video_hidden = (top_video_h * video_padding_mask.unsqueeze(-1)).sum(dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

                tgt_text_hidden = (tgt_top_text_h * tgt_text_padding_mask.unsqueeze(-1)).sum(
                    dim=1) / tgt_text_padding_mask.sum(
                    dim=1).unsqueeze(-1)
                tgt_video_hidden = (top_video_h * video_padding_mask.unsqueeze(-1)).sum(dim=1) / video_padding_mask.sum(
                    dim=1).unsqueeze(-1)

            src_text_hidden = src_text_hidden.half()
            src_video_hidden = src_video_hidden.half()

            src_text_hidden = self.src_text_projection(src_text_hidden).float()
            src_video_hidden = self.src_video_projection(src_video_hidden).float()

            src_batch_size, src_hidden_size = src_text_hidden.size()
            src_logits = F.cosine_similarity(src_text_hidden.expand((src_batch_size, src_batch_size, src_hidden_size)),
                                             src_video_hidden.expand(
                                                 (src_batch_size, src_batch_size, src_hidden_size)).transpose(0, 1),
                                             dim=-1)
            src_logits /= self.contrastive_temperature
            src_loss = -torch.nn.LogSoftmax(0)(src_logits).diag()


            tgt_text_hidden = tgt_text_hidden.half()
            tgt_video_hidden = tgt_video_hidden.half()

            tgt_text_hidden = self.tgt_text_projection(tgt_text_hidden).float()
            tgt_video_hidden = self.tgt_video_projection(tgt_video_hidden).float()

            tgt_batch_size, tgt_hidden_size = tgt_text_hidden.size()
            tgt_logits = F.cosine_similarity(tgt_text_hidden.expand((tgt_batch_size, tgt_batch_size, tgt_hidden_size)),
                                             tgt_video_hidden.expand(
                                                 (tgt_batch_size, tgt_batch_size, tgt_hidden_size)).transpose(0, 1),
                                             dim=-1)
            tgt_logits /= self.contrastive_temperature
            tgt_loss = -torch.nn.LogSoftmax(0)(tgt_logits).diag()

            if reduce:
                src_loss = src_loss.sum()
                tgt_loss = tgt_loss.sum()

        return src_loss, tgt_loss, src_text_hidden, src_video_hidden, tgt_text_hidden, tgt_video_hidden

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:

        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        src_contrastive_loss_sum = sum(log.get("src_contrastive_loss", 0) for log in logging_outputs)
        tgt_contrastive_loss_sum = sum(log.get("tgt_contrastive_loss", 0) for log in logging_outputs)

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
            "src_contrasitve_loss", src_contrastive_loss_sum / nsentences / math.log(2), nsentences, round=3
        )
        metrics.log_scalar(
            "tgt_src_contrasitve_loss", tgt_contrastive_loss_sum / nsentences / math.log(2), nsentences, round=3
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

        src2video_modal_similarity_sum = sum(log.get("src2video_modal_similarity", 0) for log in logging_outputs)
        tgt2video_modal_similarity_sum = sum(log.get("tgt2video_modal_similarity", 0) for log in logging_outputs)
        src2tgt_modal_similarity_sum = sum(log.get("src2tgt_modal_similarity", 0) for log in logging_outputs)

        metrics.log_scalar(
            "src2video_modal_similarity", src2video_modal_similarity_sum / len(logging_outputs) / GPU_nums, round=5
        )
        metrics.log_scalar(
            "tgt2video_modal_similarity", tgt2video_modal_similarity_sum / len(logging_outputs) / GPU_nums, round=5
        )
        metrics.log_scalar(
            "src2tgt_modal_similarity", src2tgt_modal_similarity_sum / len(logging_outputs) / GPU_nums, round=5
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
