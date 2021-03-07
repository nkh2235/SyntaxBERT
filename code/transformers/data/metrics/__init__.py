# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, precision_recall_curve, auc
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(pred_labels, labels):
        return (pred_labels == labels).mean()


    def acc_auc_f1(preds, pred_labels, labels):
        acc = simple_accuracy(pred_labels, labels)
        f1 = f1_score(y_true=labels, y_pred=pred_labels)
        roc_auc = roc_auc_score(y_true=labels, y_score=preds[:, 1])
        precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=preds[:, 1])
        pr_auc = auc(recall, precision)
        return {
            "acc": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }


    def pearson_and_spearman(pred_labels, labels):
        pearson_corr = pearsonr(pred_labels, labels)[0]
        spearman_corr = spearmanr(pred_labels, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def glue_compute_metrics(task_name, preds, pred_labels, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, pred_labels)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(pred_labels, labels)}
        elif task_name == "mrpc":
            return acc_auc_f1(preds, pred_labels, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(pred_labels, labels)
        elif task_name == "qqp":
            return acc_auc_f1(preds, pred_labels, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(pred_labels, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(pred_labels, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(pred_labels, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(pred_labels, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(pred_labels, labels)}
        else:
            raise KeyError(task_name)
