"""Implementation of AUPRO score based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable

import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import auc, roc
from torchmetrics.utilities.data import dim_zero_cat
from bisect import bisect

import numpy as np
from scipy.ndimage.measurements import label

import cv2
import numpy as np
import torch
from kornia.contrib import connected_components
from torch import Tensor

def connected_components_gpu(image: Tensor, num_iterations: int = 1000) -> Tensor:
    """Perform connected component labeling on GPU and remap the labels from 0 to N.

    Args:
        image (Tensor): Binary input image from which we want to extract connected components (Bx1xHxW)
        num_iterations (int): Number of iterations used in the connected component computation.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = connected_components(image, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()

def connected_components_cpu(image: Tensor) -> Tensor:
    """Connected component labeling on CPU.

    Args:
        image (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = torch.zeros_like(image)
    label_idx = 1
    for i, mask in enumerate(image):
        mask = mask.squeeze().numpy().astype(np.uint8)
        _, comps = cv2.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            components[i, 0, ...][np.where(comps == label)] = label_idx
            label_idx += 1
    return components.int()

class AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[Tensor]
    target: list[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Any | None = None,
        dist_sync_fn: Callable | None = None,
        fpr_limit: float = 0.3,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def perform_cca(self) -> Tensor:
        """Perform the Connected Component Analysis on the self.target tensor.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with requirements imposed by kornia for
                        connected component analysis.

        Returns:
            Tensor: Components labeled from 0 to N.
        """
        target = dim_zero_cat(self.target)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                f"interval was [{target.min()}, {target.max()}]."
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        if target.is_cuda:
            cca = connected_components_gpu(target)
        else:
            cca = connected_components_cpu(target)

        return cca

    def compute_pro(self, cca: Tensor, target: Tensor, preds: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the pro/fpr value-pairs until the fpr specified by self.fpr_limit.

        It leverages the fact that the overlap corresponds to the tpr, and thus computes the overall
        PRO curve by aggregating per-region tpr/fpr values produced by ROC-construction.

        Returns:
            tuple[Tensor, Tensor]: tuple containing final fpr and tpr values.
        """

        # compute the global fpr-size
        fpr: Tensor = roc(preds, target)[0]  # only need fpr
        output_size = torch.where(fpr <= self.fpr_limit)[0].size(0)

        # compute the PRO curve by aggregating per-region tpr/fpr curves/values.
        tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)

        # Loop over the labels, computing per-region tpr/fpr curves, and aggregating them.
        # Note that, since the groundtruth is different for every all to `roc`, we also get
        # different/unique tpr/fpr curves (i.e. len(_fpr_idx) is different for every call).
        # We therefore need to resample per-region curves to a fixed sampling ratio (defined above).
        labels = cca.unique()[1:]  # 0 is background
        background = cca == 0
        _fpr: Tensor
        _tpr: Tensor
        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca == label
            # Need to calculate label-wise roc on union of background & mask, as otherwise we wrongly consider other
            # label in labels as FPs. We also don't need to return the thresholds
            _fpr, _tpr = roc(preds[background | mask], mask[background | mask])[:-1]

            # catch edge-case where ROC only has fpr vals > self.fpr_limit
            if _fpr[_fpr <= self.fpr_limit].max() == 0:
                _fpr_limit = _fpr[_fpr > self.fpr_limit].min()
            else:
                _fpr_limit = self.fpr_limit

            _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
            # if computed roc curve is not specified sufficiently close to self.fpr_limit,
            # we include the closest higher tpr/fpr pair and linearly interpolate the tpr/fpr point at self.fpr_limit
            if not torch.allclose(_fpr[_fpr_idx].max(), self.fpr_limit):
                _tmp_idx = torch.searchsorted(_fpr, self.fpr_limit)
                _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
                _slope = 1 - ((_fpr[_tmp_idx] - self.fpr_limit) / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1]))
                interp = True

            _fpr = _fpr[_fpr_idx]
            _tpr = _tpr[_fpr_idx]

            _fpr_idx = _fpr_idx.float()
            _fpr_idx /= _fpr_idx.max()
            _fpr_idx *= new_idx.max()

            if interp:
                # last point will be sampled at self.fpr_limit
                new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)

            _tpr = self.interp1d(_fpr_idx, _tpr, new_idx)
            _fpr = self.interp1d(_fpr_idx, _fpr, new_idx)
            tpr += _tpr
            fpr += _fpr

        # Actually perform the averaging
        tpr /= labels.size(0)
        fpr /= labels.size(0)
        return fpr, tpr

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute the PRO curve.

        Perform the Connected Component Analysis first then compute the PRO curve.

        Returns:
            tuple[Tensor, Tensor]: tuple containing final fpr and tpr values.
        """

        cca = self.perform_cca().flatten()
        target = dim_zero_cat(self.target).flatten()
        preds = dim_zero_cat(self.preds).flatten()

        return self.compute_pro(cca=cca, target=target, preds=preds)

    def compute(self) -> Tensor:
        """Fist compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, tpr = self._compute()

        aupro = auc(fpr, tpr, reorder=True)
        aupro = aupro / fpr[-1]  # normalize the area

        return aupro

    @staticmethod
    def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
        """Function to interpolate a 1D signal linearly to new sampling points.

        Args:
            old_x (Tensor): original 1-D x values (same size as y)
            old_y (Tensor): original 1-D y values (same size as x)
            new_x (Tensor): x-values where y should be interpolated at

        Returns:
            Tensor: y-values at corresponding new_x values.
        """

        # Compute slope
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # Prepare idx for linear interpolation
        idx = torch.searchsorted(old_x, new_x)

        # searchsorted looks for the index where the values must be inserted
        # to preserve order, but we actually want the preceeding index.
        idx -= 1
        # we clamp the index, because the number of intervals = old_x.size(0) -1,
        # and the left neighbour should hence be at most number of intervals -1, i.e. old_x.size(0) - 2
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # perform actual linear interpolation
        y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])

        return y_new


    ####
    def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
        """Compute the area under the curve of per-region overlapping (PRO) and 0 to 0.3 FPR
        Args:
            masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
            amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
            num_th (int, optional): Number of thresholds
        Returns:
            float: PRO AUC (Area Under the Curve)
        """
        assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
        assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
        assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
        assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
        assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be the same"
        assert isinstance(num_th, int), "type(num_th) must be int"
        
        aupro = AUPRO(fpr_limit=0.3)
        aupro.update(amaps, masks)
        computed_aupro = aupro.compute()

        return computed_aupro