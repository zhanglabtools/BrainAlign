# -*- coding: utf-8 -*-
# @author: Xingyan Liu

from .utils import (
        load_hidden_states,
        save_hidden_states,
        load_example_data
)
from .utils import base
from .utils.base import (
        save_pickle,
        load_pickle,
        save_json_dict,
        load_json_dict,
        check_dirs,
        write_info,
        make_nowtime_tag,
        subsample_each_group,
        )
from .utils import preprocess as pp
from .utils import plot as pl
from .utils import analyze as ana
from .utils.analyze import (
        load_dpair_and_model,
        weight_linked_vars,
        make_abstracted_graph,
        )
from .utils.train import prepare4train, Trainer, SUBDIR_MODEL
from .utils._base_trainer import get_checkpoint_list
from .utils.evaluation import accuracy
from .model import (
        Predictor,
        detach2numpy,
        as_probabilities,
        predict_from_logits,
        predict,
        CGGCNet,
        CGCNet
)
from .datapair import (
        datapair_from_adatas,
        aligned_datapair_from_adatas,
        DataPair,
        AlignedDataPair,
        make_features,
)
from .PARAMETERS import get_model_params, get_loss_params
from . import pipeline
from .pipeline import KET_CLUSTER, __test1__, __test2__


__version__ = "0.1.8"
