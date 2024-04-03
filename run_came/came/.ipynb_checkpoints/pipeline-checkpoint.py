# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Sequence, Union, Mapping, List, Optional  # , Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import sparse
import torch
import logging
from . import (
    pp, pl,
    save_pickle,
    detach2numpy,
    save_json_dict,
    make_nowtime_tag,
    write_info,
    as_probabilities,
    predict_from_logits,
    subsample_each_group,
    SUBDIR_MODEL,
)
from .PARAMETERS import (
    get_model_params,
    get_loss_params,
    # get_preprocess_params
)
from . import (
    Predictor, DataPair, AlignedDataPair,
    CGGCNet, datapair_from_adatas,
    CGCNet, aligned_datapair_from_adatas
)
from .utils.train import prepare4train, Trainer

PARAMS_MODEL = get_model_params()
# PARAMS_PRE = get_preprocess_params()
PARAMS_LOSS = get_loss_params()

KET_CLUSTER = 'clust_lbs'


def main_for_aligned(
        adatas: Sequence[sc.AnnData],
        vars_feat: Sequence,
        vars_as_nodes: Optional[Sequence] = None,
        scnets: Optional[Sequence[sparse.spmatrix]] = None,
        dataset_names: Sequence[str] = ('reference', 'query'),
        key_class1: str = 'cell_ontology_class',
        key_class2: Optional[str] = None,
        do_normalize: Union[bool, Sequence[bool]] = True,
        batch_keys=None,
        n_epochs: int = 350,
        resdir: Union[Path, str] = None,
        tag_data: Optional[str] = None,  # for autometically deciding `resdir` for results saving
        params_model: dict = {},
        params_lossfunc: dict = {},
        n_pass: int = 100,
        batch_size: Optional[int] = None,
        pred_batch_size: Union[int, str, None] = 'auto',
        plot_results: bool = False,
        norm_target_sum: Optional[float] = None,
        save_hidden_list: bool = True,
        save_dpair: bool = True,
        device_id = '0',
):
    """ Run the main process of CAME (model training), for integrating 2 datasets
    of aligned features. (e.g., cross-species integration)

    Parameters
    ----------

    adatas
        A pair of ``sc.AnnData`` objects, the reference and query raw data
    vars_feat: a sequence of strings
        variables to be taken as the node-features of the observations
    vars_as_nodes: a sequence of strings
        variables to be taken as the graph nodes
    scnets
        two single-cell-networks or a merged one
    dataset_names
        a tuple of two names for reference and query, respectively
    key_class1
        the key to the type-labels for the reference data,
        should be a column name of ``adatas[0].obs``.
    key_class2
        the key to the type-labels for the query data. Optional, if provided,
        should be a column name of ``adatas[1].obs``.
    do_normalize
        whether to normalize the input data
        (the they have already been normalized, set it False)
    batch_keys
        a list of two strings (or None), specifying the batch-keys for
        data1 and data2, respectively.
        if given, features (of cell nodes) will be scaled within each batch.
    n_epochs
        number of training epochs.
        A recommended setting is 200-400 for whole-graph training,
        and 80-200 for sub-graph training.
    resdir
        directory for saving results output by CAME
    tag_data
        a tag for auto-creating result directory
    params_model
        the model parameters
    params_lossfunc
        parameters for loss function
    n_pass
        number of epochs to skip; not backup model checkpoints until ``n_pass``
        epochs.
    batch_size
        the number of observation nodes in each mini-batch, based on which the
        sub-graphs will be used for mini-batch training.
        if None, the model will be trained on the whole graph.
    pred_batch_size
        batch-size in prediction process
    plot_results
        whether to automatically plot the classification results
    norm_target_sum
        the scale factor for library-size normalization
    save_hidden_list
        whether to save the hidden states for all the layers
    save_dpair
        whether to save the elements of the DataPair object

    Returns
    -------
    outputs: dict
    """
    if resdir is None:
        tag_time = make_nowtime_tag()
        tag_data = dataset_names if tag_data is None else tag_data
        resdir = Path(f'{tag_data}-{tag_time}')
    else:
        resdir = Path(resdir)

    figdir = resdir / 'figs'
    pp.check_dirs(figdir)
    sc.settings.figdir = figdir
    pp.check_dirs(resdir)
    # keys for training process
    if key_class2 is None:
        if key_class1 in adatas[1].obs.columns:
            key_class2 = key_class1
            keys = [key_class1, key_class2]
        else:
            key_class2 = KET_CLUSTER
            keys = [key_class1, None]
    else:
        keys = [key_class1, key_class2]
    keys_compare = [key_class1, key_class2]
    adatas = list(adatas)
    if isinstance(do_normalize, bool):
        do_normalize = [do_normalize] * 2
    if do_normalize[0]:
        adatas[0] = pp.normalize_default(
                adatas[0], target_sum=norm_target_sum, force_return=True)
    if do_normalize[1]:
        adatas[1] = pp.normalize_default(
                adatas[1], target_sum=norm_target_sum, force_return=True)

        # if do_normalize:
        # adatas = list(map(
        #     lambda a: pp.normalize_default(
        #         a, target_sum=norm_target_sum, force_return=True),
        #     adatas))

    logging.info('Step 1: preparing AlignedDataPair object...')
    dpair = aligned_datapair_from_adatas(
        adatas,
        vars_feat=vars_feat,
        vars_as_nodes=vars_as_nodes,
        oo_adjs=scnets,
        dataset_names=dataset_names,
    )
    print(dpair)

    ENV_VARs = prepare4train(dpair, key_class=keys, batch_keys=batch_keys)

    logging.debug(ENV_VARs.keys())
    g = ENV_VARs['g']
    classes = ENV_VARs['classes']
    # classes = classes0[:-1] if 'unknown' in classes0 else classes0
    n_classes = len(classes)
    params_model = get_model_params(**params_model)
    params_model.update(
        g_or_canonical_etypes=g.canonical_etypes,
        in_dim_dict={'cell': dpair.n_feats, 'gene': 0},
        out_dim=n_classes,
        layernorm_ntypes=g.ntypes
    )
    save_json_dict(params_model, resdir / 'model_params.json')

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGCNet(**params_model)

    params_lossfunc = get_loss_params(**params_lossfunc)
    trainer = Trainer(model=model, dir_main=resdir, **ENV_VARs)

    if batch_size is not None:
        trainer.train_minibatch(
            n_epochs=n_epochs,
            params_lossfunc=params_lossfunc,
            batch_size=batch_size,
            n_pass=n_pass, device=device)
    else:
        trainer.train(n_epochs=n_epochs,
                      params_lossfunc=params_lossfunc,
                      n_pass=n_pass, device=device)
    trainer.save_model_weights()
    # ========================== record results ========================
    trainer.write_train_logs()

    write_info(resdir / 'info.txt',
               current_performance=trainer._cur_log,
               params_model=params_model,
               graph=g,
               model=model,
               )
    trainer.plot_cluster_index(fp=figdir / 'cluster_index.png')

    # ======================== Gather results ======================
    if pred_batch_size == 'auto':
        pred_batch_size = batch_size
    outputs = gather_came_results(
        dpair,
        trainer,
        classes=classes,
        keys=keys,
        keys_compare=keys_compare,
        resdir=resdir,
        checkpoint='best',
        batch_size=pred_batch_size,
        save_hidden_list=save_hidden_list,
        save_dpair=save_dpair,
    )
    # ============= confusion matrix & heatmap plot ==============
    if plot_results:
        obs = dpair.obs
        obs_ids1 = detach2numpy(trainer.train_idx)
        obs_ids2 = detach2numpy(trainer.test_idx)
        test_acc = trainer.test_acc[trainer._cur_epoch_adopted]
        if key_class2 == KET_CLUSTER:
            # labels_cat = obs['celltype']
            key_true = 'celltype'
            acc_tag = ''
        else:
            # labels_cat = obs["REF"]
            key_true = 'REF'
            acc_tag = f'(acc{test_acc:.1%})'
        # cl_preds = obs['predicted']
        try:
            plot_class_results(
                obs, obs_ids1, obs_ids2,
                key_true=key_true, key_pred='predicted',
                figdir=figdir, acc_tag=acc_tag, classes=classes,
                df_probs=outputs['df_probs']
            )
        except Exception as e:
            logging.warning(f'An error occurred when plotting results: {e}')

    return outputs


def main_for_unaligned(
        adatas: Sequence[sc.AnnData],
        vars_feat: Sequence[Sequence],
        vars_as_nodes: Sequence[Sequence],
        df_varmap: pd.DataFrame,
        df_varmap_1v1: Optional[pd.DataFrame] = None,
        scnets: Optional[Sequence[sparse.spmatrix]] = None,
        union_var_nodes: bool = True,
        union_node_feats: bool = True,
        keep_non1v1_feats: bool = False,
        col_weight: Optional[str] = None,
        non1v1_trans_to: int = 0,  # should only be in {0, 1}
        dataset_names: Sequence[str] = ('reference', 'query'),
        key_class1: str = 'cell_ontology_class',
        key_class2: Optional[str] = None,
        do_normalize: bool = True,
        batch_keys=None,
        n_epochs: int = 350,
        resdir: Union[Path, str] = None,
        tag_data: Optional[str] = None,
        params_model: dict = {},
        params_lossfunc: dict = {},
        n_pass: int = 100,
        batch_size: Optional[int] = None,
        pred_batch_size: Union[int, str, None] = 'auto',
        plot_results: bool = False,
        norm_target_sum: Optional[float] = None,
        save_hidden_list: bool = True,
        save_dpair: bool = True,
        device_id = '0',
        h_dim=128):
    """ Run the main process of CAME (model training), for integrating 2 datasets
    of unaligned features. (e.g., cross-species integration)

    Parameters
    ----------
    adatas
        A pair of ``sc.AnnData`` objects, the reference and query raw data
    vars_feat
        A list or tuple of 2 variable name-lists.
        for example, differential expressed genes, highly variable features.
    vars_as_nodes: list or tuple of 2
        variables to be taken as the graph nodes
    df_varmap
        A ``pd.DataFrame`` with (at least) 2 columns; required.
        relationships between features in 2 datasets, for making the
        adjacent matrix (`vv_adj`) between variables from these 2 datasets.
    df_varmap_1v1: None, pd.DataFrame; optional.
        dataframe containing only 1-to-1 correspondence between features
        in 2 datasets, if not provided, it will be inferred from `df_varmap`
    scnets
        two single-cell-networks or a merged one
    union_var_nodes: bool
        whether to take the union of the variable-nodes
    union_node_feats: bool
        whether to take the union of the observation(cell)-node features
    keep_non1v1_feats: bool
        whether to take into account the non-1v1 variables as the node features.
        If most of the homologies are non-1v1, better set this as True!
    col_weight
        A column in ``df_varmap`` specifying the weights between homologies.
    non1v1_trans_to: int
        the direction to transform non-1v1 features, should either be 0 or 1.
        Set as 0 to transform query data to the reference (default),
        1 to transform the reference data to the query.
        If set ``keep_non1v1_feats=False``, this parameter will be ignored.
    dataset_names
        a tuple of two names for reference and query, respectively
    key_class1
        the key to the type-labels for the reference data,
        should be a column name of ``adatas[0].obs``.
    key_class2
        the key to the type-labels for the query data. Optional, if provided,
        should be a column name of ``adatas[1].obs``.
    do_normalize
        whether to normalize the input data
        (the they have already been normalized, set it False)
    batch_keys
        a list of two strings (or None), specifying the batch-keys for
        data1 and data2, respectively.
        if given, features (of cell nodes) will be scaled within each batch
    n_epochs
        number of training epochs.
        A recommended setting is 200-400 for whole-graph training,
        and 80-200 for sub-graph training.
    resdir
        directory for saving results output by CAME
    tag_data
        a tag for auto-creating the result directory ``resdir``
    params_model
        the model parameters
    params_lossfunc
        parameters for loss function
    n_pass
        number of epochs to skip; not backup model checkpoints until ``n_pass``
        epochs.
    batch_size
        the number of observation nodes in each mini-batch, based on which the
        sub-graphs will be used for mini-batch training.
        if None, the model will be trained on the whole graph.
    pred_batch_size
        batch-size in prediction process
    plot_results
        whether to automatically plot the classification results
    norm_target_sum
        the scale factor for library-size normalization
    save_hidden_list
        whether to save the hidden states for all the layers
    save_dpair
        whether to save the elements of the DataPair object

    Returns
    -------
    outputs: dict
    """
    if resdir is None:
        tag_time = make_nowtime_tag()
        tag_data = dataset_names if tag_data is None else tag_data
        resdir = Path(f'{tag_data}-{tag_time}')
    else:
        resdir = Path(resdir)

    figdir = resdir / 'figs'
    pp.check_dirs(figdir)
    sc.settings.figdir = figdir
    pp.check_dirs(resdir)
    # keys for training process
    if key_class2 is None:
        if key_class1 in adatas[1].obs.columns:
            key_class2 = key_class1
            keys = [key_class1, key_class2]
        else:
            key_class2 = KET_CLUSTER
            keys = [key_class1, None]
    else:
        keys = [key_class1, key_class2]
    keys_compare = [key_class1, key_class2]
    adatas = list(adatas)

        
    #label_list  = [x for x in adatas[0].obs[key_class1]] + [x for x in adatas[1].obs[key_class2]]
    
    
    if isinstance(do_normalize, bool):
        do_normalize = [do_normalize] * 2
    if do_normalize[0]:
        adatas[0] = pp.normalize_default(
                adatas[0], target_sum=norm_target_sum, force_return=True)
    if do_normalize[1]:
        adatas[1] = pp.normalize_default(
                adatas[1], target_sum=norm_target_sum, force_return=True)
    # if do_normalize:
    #     adatas = list(map(
    #         lambda a: pp.normalize_default(
    #             a, target_sum=norm_target_sum, force_return=True),
    #         adatas
    #     ))
    logging.info('preparing DataPair object...')
    dpair = datapair_from_adatas(adatas, vars_feat=vars_feat,
                                 df_varmap=df_varmap,
                                 df_varmap_1v1=df_varmap_1v1, oo_adjs=scnets,
                                 vars_as_nodes=vars_as_nodes,
                                 union_var_nodes=union_var_nodes,
                                 union_node_feats=union_node_feats,
                                 dataset_names=dataset_names,
                                 keep_non1v1_feats=keep_non1v1_feats,
                                 non1v1_trans_to=non1v1_trans_to,
                                 col_weight=col_weight,
                                 )
    print(dpair)

    ENV_VARs = prepare4train(dpair, key_class=keys, batch_keys=batch_keys,)

    logging.info(ENV_VARs.keys())
    g = ENV_VARs['g']
    classes = ENV_VARs['classes']
    # classes = classes0[:-1] if 'unknown' in classes0 else classes0
    n_classes = len(classes)
    params_model = get_model_params(**params_model)
    params_model.update(
        g_or_canonical_etypes=g.canonical_etypes,
        in_dim_dict={'cell': dpair.n_feats, 'gene': 0},
        out_dim=n_classes,
        layernorm_ntypes=g.ntypes,
        h_dim=h_dim
    )
    save_json_dict(params_model, resdir / 'model_params.json')

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGGCNet(**params_model)

    # training
    params_lossfunc = get_loss_params(**params_lossfunc)
    trainer = Trainer(model=model, dir_main=resdir, **ENV_VARs)
    if batch_size is not None:
        trainer.train_minibatch_cluster(
            n_epochs=n_epochs,
            params_lossfunc=params_lossfunc,
            batch_size=batch_size,
            n_pass=n_pass, device=device)
        # trainer.train_minibatch(
        #     n_epochs=n_epochs,
        #     params_lossfunc=params_lossfunc,
        #     batch_size=batch_size,
        #     n_pass=n_pass, device=device)
    else:
        trainer.train(n_epochs=n_epochs,
                      params_lossfunc=params_lossfunc,
                      n_pass=n_pass, device=device)
    trainer.save_model_weights()
    # ========================== record results ========================
    trainer.write_train_logs()
    write_info(resdir / 'info.txt',
               current_performance=trainer._cur_log,
               params_model=params_model,
               graph=g,
               model=model,
               )
    trainer.plot_cluster_index(fp=figdir / 'cluster_index.png')

    # ======================== Gather results ======================
    if pred_batch_size == 'auto':
        pred_batch_size = batch_size
    # out_cell, df_probs, h_dict, predictor = gather_came_results(
    outputs = gather_came_results(
        dpair,
        trainer,
        classes=classes,
        keys=keys,
        keys_compare=keys_compare,
        resdir=resdir,
        checkpoint='best',
        batch_size=pred_batch_size,
        save_hidden_list=save_hidden_list,
        save_dpair=save_dpair,
    )
    if plot_results:
        obs = dpair.obs
        obs_ids1 = detach2numpy(trainer.train_idx)
        obs_ids2 = detach2numpy(trainer.test_idx)
        test_acc = trainer.test_acc[trainer._cur_epoch_adopted]
        if key_class2 == KET_CLUSTER:
            # labels_cat = obs['celltype']
            key_true = 'celltype'
            acc_tag = ''
        else:
            # labels_cat = obs["REF"]
            key_true = 'REF'
            acc_tag = f'(acc{test_acc:.1%})'
        # cl_preds = obs['predicted']
        try:
            plot_class_results(
                obs, obs_ids1, obs_ids2,
                key_true=key_true, key_pred='predicted',
                figdir=figdir, acc_tag=acc_tag, classes=classes,
                df_probs=outputs['df_probs']
            )
        except Exception as e:
            logging.warning(f'An error occurred when plotting results: {e}')

    return outputs


def plot_class_results(
        obs, obs_ids1, obs_ids2,
        key_true='REF', key_pred='predicted',
        figdir=Path('.'),
        acc_tag: str = '',
        classes: Sequence = None,
        df_probs: pd.DataFrame = None,
):
    # obs_ids1 = detach2numpy(trainer.train_idx)
    # obs_ids2 = detach2numpy(trainer.test_idx)
    # test_acc = trainer.test_acc[trainer._cur_epoch_adopted]
    # if key_class2 == KET_CLUSTER:
    #     labels_cat = obs['celltype']
    #     acc_tag = ''
    # else:
    #     labels_cat = obs["REF"]
    #     acc_tag = f'(acc{test_acc:.1%})'
    labels_cat = obs[key_true]
    cl_preds = obs[key_pred]

    # confusion matrix OR alluvial plot
    sc.set_figure_params(fontsize=10)
    ax, contmat = pl.plot_contingency_mat(
        labels_cat[obs_ids2], cl_preds[obs_ids2], norm_axis=1,
        fp=figdir / f'contingency_matrix{acc_tag}.png',
    )
    pl.plot_confus_mat(
        labels_cat[obs_ids1], cl_preds[obs_ids1], classes_on=classes,
        fp=figdir / f'contingency_matrix-train.png',
    )

    # heatmap of predicted probabilities
    # df_probs = outputs['df_probs']
    if df_probs is not None:
        gs = pl.wrapper_heatmap_scores(
            df_probs.iloc[obs_ids2], obs.iloc[obs_ids2], ignore_index=True,
            col_label='celltype', col_pred='predicted',
            n_subsample=50,
            cmap_heat='magma_r',  # if prob_func == 'softmax' else 'RdBu_r'
            fp=figdir / f'heatmap_probas.pdf'
        )


def gather_came_results(
        dpair: Union[DataPair, AlignedDataPair],
        trainer: Trainer,
        classes: Sequence,
        keys: Sequence[str],
        keys_compare: Optional[Sequence[str]] = None,
        resdir: Union[str, Path] = '.',
        checkpoint: Union[int, str] = 'best',
        batch_size: Optional[int] = None,
        save_hidden_list: bool = True,
        save_dpair: bool = True,
):
    """ Packed function for pipeline as follows:

    1. load the 'best' or the given checkpoint (model)
    2. get the predictions for cells, including probabilities (from logits)
    3. get and the hidden states for both cells and genes
    4. make a predictor

    Parameters
    ----------
    dpair
        the ``DataPair`` or ``AlignedDataPair`` object. Note that it may be changed
        after pass through this function.
    trainer
        the model trainer
    classes
        the class (or cell-type) space
    keys
        a pair of names like [`key_class1`, `key_class2`], where `key_class1`
        is the column name of the reference cell-type labels, and
        `key_class2` for the query, which can be set as None if there
        are no labels for the query data.
        These labels will be extracted and stored in the column 'REF' of
        ``dpair.obs``.
    keys_compare
        a pair of names like [key_class1, key_class2], just for comparison.
        These labels will be extracted and stored in the column 'celltype'
        of ``dpair.obs``.
    resdir
        the result directory
    checkpoint
        specify which checkpoint to adopt.
        candidates are 'best', 'last', or an integer.
    batch_size
        specify it when your GPU memory is limited
    save_hidden_list
        whether to save the hidden states into `{resdir}/hidden_list.h5`
    save_dpair
        whether to save the dpair elements into `{resdir}/datapair_init.pickle`

    """
    resdir = Path(resdir)
    keys_compare = keys if keys_compare is None else keys_compare
    if isinstance(checkpoint, int):
        trainer.load_model_weights(checkpoint)
    elif 'best' in checkpoint.lower():
        trainer.load_model_weights()
    elif 'last' in checkpoint.lower():
        trainer.load_model_weights(trainer._cur_epoch)
    else:
        raise ValueError(
            f'`checkpoint` should be either str ("best" or "last") or int, '
            f'got {checkpoint}'
        )
    # all hidden states
    from .model import get_all_hidden_states, get_attentions
    hidden_list = get_all_hidden_states(
        trainer.model, trainer.feat_dict, trainer.g, batch_size=batch_size)
    if save_hidden_list:
        from . import save_hidden_states
        save_hidden_states(hidden_list, resdir / 'hidden_list.h5')
    # hidden states are stored to facilitated downstream analysis
    h_dict = hidden_list[-1]
    # if batch_size:
    #     out_cell = trainer.get_current_outputs(batch_size=batch_size)['cell']
    # else:
    # trainer.model.eval()
    # out_cell = trainer.model.cell_classifier.forward(
    #     trainer.g,
    #     {k: torch.Tensor(h).to(trainer.g.device) for k, h in h_dict.items()}
    # )['cell']
    attn_mat, out_cell = get_attentions(
        trainer.model,
        {k: torch.Tensor(h) for k, h in h_dict.items()},
        trainer.g, from_scratch=False, is_train=False, return_logits=True,
        device=trainer.g.device
    )
    sparse.save_npz(f"{resdir}/attentions.npz", attn_mat)
    out_cell = detach2numpy(out_cell)

    pd.DataFrame(out_cell[dpair.obs_ids1], columns=classes).to_csv(resdir / "df_logits1.csv")
    pd.DataFrame(out_cell[dpair.obs_ids2], columns=classes).to_csv(resdir / "df_logits2.csv")
    predictor = Predictor(classes=classes).fit(
            out_cell[detach2numpy(trainer.train_idx)],
            detach2numpy(trainer.train_labels),
        )
    predictor.save(resdir / f'predictor.json')

    probas_all = as_probabilities(out_cell, mode='softmax')
    cl_preds = predict_from_logits(probas_all, classes=classes)
    obs = pd.DataFrame(
        # TODO: `keys[0]` -> 'REF', may raise another exceptions
        {'REF': dpair.get_obs_labels(keys, asint=False, categories=classes),
         # true labels with `unknown` for unseen classes in query data
         'celltype': dpair.get_obs_anno(keys_compare),  # labels for comparison
         'predicted': cl_preds,
         'max_probs': np.max(probas_all, axis=1),
         })
    obs['is_right'] = obs['predicted'] == obs['REF']
    df_probs = pd.DataFrame(probas_all, columns=classes)
    dpair.set_common_obs_annos(obs)
    dpair.set_common_obs_annos(df_probs, ignore_index=True)
    dpair.obs.to_csv(resdir / 'obs.csv')
    if save_dpair:
        dpair.save_init(resdir / 'datapair_init.pickle')

    # group counts statistics (optional)
    gcnt = pp.group_value_counts(dpair.obs, 'celltype', group_by='dataset')
    logging.info(str(gcnt))
    gcnt.to_csv(resdir / 'group_counts.csv')
    # return out_cell, df_probs, h_dict, predictor
    outputs = {
        "dpair": dpair,
        "trainer": trainer,
        "h_dict": h_dict,
        "out_cell": out_cell,
        "predictor": predictor,
        "df_probs": df_probs,
        "attentions": attn_mat,
    }
    return outputs


def preprocess_aligned(
        adatas: [sc.AnnData, sc.AnnData],
        key_class: str,
        df_varmap_1v1: Optional[pd.DataFrame] = None,
        use_scnets: bool = True,
        n_pcs: int = 30,
        nneigh_scnet: int = 5,
        nneigh_clust: int = 20,
        deg_cuts: dict = {},
        ntop_deg: Optional[int] = 50,
        ntop_deg_nodes: Optional[int] = 50,
        key_clust: str = KET_CLUSTER,  # 'clust_lbs'
        node_source: str = 'hvg,deg',
        ext_feats: Optional[Sequence] = None,
        ext_nodes: Optional[Sequence] = None,
        n_top_genes: int = 2000,
):
    """
    Packed function for process adatas with aligned features
    (i.e., one-to-one correspondence).

    Processing Steps:

        * align variables
        * preprocessing
        * candidate genes (HVGs and DEGs)
        * pre-clustering query data
        * computing single-cell network

    Parameters
    ----------

    adatas
        A pair of ``sc.AnnData`` objects, the reference and query raw data
    key_class
        the key to the type-labels, should be a column name of ``adatas[0].obs``
    df_varmap_1v1
        dataframe containing only 1-to-1 correspondence between features
        in ``adatas``; if not provided, map the variables of their original names.
    use_scnets
        whether to use the cell-cell-similarity edges (single-cell-network)
    n_pcs
        the number of PCs for computing the single-cell-network
    nneigh_scnet
        the number of nearest neighbors to account for the single-cell-network
    nneigh_clust
        the number of nearest neighbors to account for pre-clustering
    deg_cuts
        dict with keys 'cut_padj', 'cut_pts', and 'cut_logfc', used for
        filtering DEGs.
    ntop_deg
        the number of top DEGs to take as the node-features
    ntop_deg_nodes
        the number of top DEGs to take as the graph nodes
    key_clust
        where to add the per-clustering labels to the query data, i.e.,
        ``adatas[1].obs``. By default, it's set as ``came.pipeline.KEY_CLUSTER``
    node_source
        source of the node genes, using both DEGs and HVGs by default
    ext_feats
        extra variables (genes) to be added to the auto-selected ones as the
        **observation(cell)-node features**.
    ext_nodes
        extra variables (genes) to be added to the auto-selected ones as the
        **variable(gene)-nodes**.

    Returns
    -------
        came_inputs: a dict containing CAME inputs
        (adata1, adata2): a tuple of the preprocessed ``AnnData`` objects
    """
    adatas = pp.align_adata_vars(
        adatas[0], adatas[1], df_varmap_1v1, unify_names=True,
    )

    logging.info('================ preprocessing ===============')
    params_preproc = dict(
        target_sum=None,
        n_top_genes=n_top_genes,#500,#2000,
        n_pcs=n_pcs,
        nneigh=nneigh_scnet,
        copy=True,
    )
    # NOTE: using the median total-counts as the scale factor
    # (may perform better than fixed number)
    adata1 = pp.quick_preprocess(adatas[0], **params_preproc)
    adata2 = pp.quick_preprocess(adatas[1], **params_preproc)

    # the single-cell network
    if use_scnets:
        scnets = [pp.get_scnet(adata1), pp.get_scnet(adata2)]
    else:
        scnets = None
    # get HVGs
    hvgs1, hvgs2 = pp.get_hvgs(adata1), pp.get_hvgs(adata2)

    # cluster labels
    clust_lbs2 = pp.get_leiden_labels(
        adata2, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[1].obs[key_clust] = clust_lbs2
    # DEGs
    params_deg = dict(
        n=ntop_deg, cuts=deg_cuts,
        force_redo=False, inplace=True, do_normalize=False,
        return_info=True,
    )
    # adata1&2 have already been normalized before
    deg_info1 = pp.compute_and_get_DEGs(
        adata1, key_class, **params_deg)
    deg_info2 = pp.compute_and_get_DEGs(
        adata2, key_clust, **params_deg)

    if ext_feats is None:
        ext_feats = set()
    vars_feat = sorted(set(
        pp.top_markers_from_info(deg_info1, ntop_deg)).union(
        pp.top_markers_from_info(deg_info2, ntop_deg)).union(
        ext_feats))
    # params_deg = dict(n=ntop_deg, force_redo=False,
    #                   inplace=True, do_normalize=False)
    # # adata1&2 have already been normalized before
    # degs1 = pp.compute_and_get_DEGs(
    #     adata1, key_class, **params_deg)
    # degs2 = pp.compute_and_get_DEGs(
    #     adata2, key_clust, **params_deg)
    # ###
    # vars_feat = list(set(degs1).union(degs2))
    if ext_nodes is None:
        vars_node = set()
    else:
        vars_node = set(ext_nodes)
    node_source = node_source.lower()
    if 'deg' in node_source:
        degs_nd1 = pp.top_markers_from_info(deg_info1, ntop_deg_nodes)
        degs_nd2 = pp.top_markers_from_info(deg_info2, ntop_deg_nodes)
        vars_node.update(degs_nd1)
        vars_node.update(degs_nd2)
    if 'hvg' in node_source:
        vars_node.update(hvgs1)
        vars_node.update(hvgs2)
    vars_node = sorted(vars_node)
    # if 'hvg' in node_source and 'deg' in node_source:
    #     vars_node = sorted(set(hvgs1).union(hvgs2).union(vars_feat))
    # elif 'hvg' in node_source:
    #     vars_node = sorted(set(hvgs1).union(hvgs2))
    # else:
    #     vars_node = vars_feat

    dct = dict(
        adatas=adatas,
        vars_feat=vars_feat,
        vars_as_nodes=vars_node,
        scnets=scnets,
    )
    return dct, (adata1, adata2)


def preprocess_unaligned(
        adatas: [sc.AnnData, sc.AnnData],
        key_class: str,
        use_scnets: bool = True,
        n_pcs: int = 30,
        nneigh_scnet: int = 5,
        nneigh_clust: int = 20,
        deg_cuts: dict = {},
        ntop_deg: Optional[int] = 50,
        ntop_deg_nodes: Optional[int] = 50,
        key_clust: str = KET_CLUSTER,  # 'clust_lbs'
        node_source: str = 'hvg,deg',
        ext_feats: Optional[Sequence[Sequence]] = None,
        ext_nodes: Optional[Sequence[Sequence]] = None,
        n_top_genes: int = 2000,
        do_normalize: list = [True, True],
):
    """
    Packed function for process adatas with un-aligned features.
    (i.e., some of them could be one-to-many or many-to-one correspondence)

    Processing Steps:

        * preprocessing
        * candidate genes (HVGs and DEGs)
        * pre-clustering query data
        * computing single-cell network

    Parameters
    ----------
    adatas
        A pair of ``sc.AnnData`` objects, the reference and query raw data
    key_class
        the key to the type-labels, should be a column name of ``adatas[0].obs``
    use_scnets
        whether to use the cell-cell-similarity edges (single-cell-network)
    n_pcs
        the number of PCs for computing the single-cell-network
    nneigh_scnet
        the number of nearest neighbors to account for the single-cell-network
    nneigh_clust
        the number of nearest neighbors to account for pre-clustering
    deg_cuts
        dict with keys 'cut_padj', 'cut_pts', and 'cut_logfc'
    ntop_deg
        the number of top DEGs to take as the node-features
    ntop_deg_nodes
        the number of top DEGs to take as the graph nodes, which can be
        directly displayed on the UMAP plot.
    key_clust
        where to add the per-clustering labels to the query data, i.e.,
        ``adatas[1].obs``
    node_source
        source of the node genes, using both DEGs and HVGs by default
    ext_feats
        A tuple of two lists of variable names.
        Extra variables (genes) to be added to the auto-selected ones as the
         **observation(cell)-node features**.
    ext_nodes
        A tuple of two lists of variable names.
        Extra variables (genes) to be added to the auto-selected ones as the
        **variable(gene)-nodes**.

    Returns
    -------
        came_inputs: a dict containing CAME inputs
        (adata1, adata2): a tuple of the preprocessed ``AnnData`` objects
    """
    logging.info('================ preprocessing ===============')
    params_preproc = dict(
        target_sum=None,
        n_top_genes=n_top_genes, #500, # 2000
        n_pcs=n_pcs,
        nneigh=nneigh_scnet,
        copy=True,
    )
    # NOTE:
    # by default, the original adatas will not be changed;
    # using the median total-counts as the scale factor \
    # may perform better than fixed number
    adata1 = pp.quick_preprocess(adatas[0], normalize_data = do_normalize[0], **params_preproc)
    adata2 = pp.quick_preprocess(adatas[1], normalize_data = do_normalize[1], **params_preproc)
    # the single-cell network
    if use_scnets:
        scnets = [pp.get_scnet(adata1), pp.get_scnet(adata2)]
    else:
        scnets = None
    # get HVGs
    hvgs1, hvgs2 = pp.get_hvgs(adata1), pp.get_hvgs(adata2)

    # cluster labels
    clust_lbs1 = pp.get_leiden_labels(
        adata1, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[0].obs[key_clust] = clust_lbs1

    # cluster labels
    clust_lbs2 = pp.get_leiden_labels(
        adata2, force_redo=True,
        nneigh=nneigh_clust,
        neighbors_key='clust',
        key_added=key_clust,
        copy=False
    )
    adatas[1].obs[key_clust] = clust_lbs2

    # DEGs
    params_deg = dict(
        n=ntop_deg, cuts=deg_cuts,
        force_redo=False, inplace=True, do_normalize=False,
        return_info=True,
    )
    # adata1&2 have already been normalized before
    deg_info1 = pp.compute_and_get_DEGs(adata1, 'region_name', **params_deg)
    deg_info1_clust = pp.compute_and_get_DEGs(adata1, key_clust, **params_deg)
    deg_info2 = pp.compute_and_get_DEGs(adata2, 'region_name', **params_deg)
    deg_info2_clust = pp.compute_and_get_DEGs(adata2, key_clust, **params_deg)
    # ======== Node-Features ========
    if ext_feats is None:
        vars_feat1, vars_feat2 = set(), set()
    else:
        vars_feat1, vars_feat2 = set(ext_feats[0]), set(ext_feats[1])
    vars_feat1.update(pp.top_markers_from_info(deg_info1, ntop_deg))
    vars_feat1.update(pp.top_markers_from_info(deg_info1_clust, ntop_deg))
    vars_feat2.update(pp.top_markers_from_info(deg_info2, ntop_deg))
    vars_feat2.update(pp.top_markers_from_info(deg_info2_clust, ntop_deg))
    vars_feat = [sorted(vars_feat1), sorted(vars_feat2)]
    # vars_feat = [pp.top_markers_from_info(deg_info1, ntop_deg),
    #              pp.top_markers_from_info(deg_info2, ntop_deg), ]
    # ======== VAR-Nodes ========
    if ext_nodes is None:
        nodes1, nodes2 = set(), set()
    else:
        nodes1, nodes2 = set(ext_nodes[0]), set(ext_nodes[1])
    node_source = node_source.lower()
    if 'deg' in node_source:
        nodes1.update(pp.top_markers_from_info(deg_info1, ntop_deg_nodes))
        nodes2.update(pp.top_markers_from_info(deg_info2, ntop_deg_nodes))
    if 'hvg' in node_source:
        nodes1.update(hvgs1)
        nodes2.update(hvgs2)
    vars_as_nodes = [sorted(nodes1), sorted(nodes2)]
    # ========== V1
    # if 'deg' in node_source:
    #     degs_nd1 = pp.top_markers_from_info(deg_info1, ntop_deg_nodes)
    #     degs_nd2 = pp.top_markers_from_info(deg_info2, ntop_deg_nodes)
    #     if 'hvg' in node_source:
    #         vars_as_nodes = [
    #             np.unique(np.hstack([hvgs1, degs_nd1])),
    #             np.unique(np.hstack([hvgs2, degs_nd2]))]
    #     else:
    #         vars_as_nodes = [degs_nd1, degs_nd2]
    # else:
    #     vars_as_nodes = [hvgs1, hvgs2]
    # ========= V0
    # if 'hvg' in node_source and 'deg' in node_source:
    #     vars_as_nodes = [np.unique(np.hstack([hvgs1, degs1])),
    #                      np.unique(np.hstack([hvgs2, degs2]))]
    # elif 'hvg' in node_source:
    #     vars_as_nodes = [hvgs1, hvgs2]
    # else:
    #     vars_as_nodes = [degs1, degs2]

    dct = dict(
        adatas=adatas,
        vars_feat=vars_feat,
        vars_as_nodes=vars_as_nodes,
        scnets=scnets,
    )
    return dct, (adata1, adata2)


def __test1__(n_epochs: int = 5, batch_size=None, reverse=False):
    from .utils.train import seed_everything
    seed_everything()
    datadir = Path(os.path.abspath(__file__)).parent / 'sample_data'
    if not datadir.exists():
        from .utils._get_example_data import _extract_zip
        _extract_zip()

    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')
    if reverse:
        sp1, sp2 = sp2, sp1
        dsnames = dsnames[::-1]

    df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )

    dsn1, dsn2 = dsnames
    fp1, fp2 = datadir / f'raw-{dsn1}.h5ad', datadir / f'raw-{dsn2}.h5ad'
    if not fp1.exists():
        fp1 = datadir / f'raw-{dsn1}-sampled.h5ad'

    adata_raw1, adata_raw2 = sc.read_h5ad(fp1), sc.read_h5ad(fp2)
    adatas = [adata_raw1, adata_raw2]

    key_class = 'cell_ontology_class'
    time_tag = make_nowtime_tag()
    resdir = Path('_temp') / f'{dsnames}-{time_tag}'

    came_inputs, (adata1, adata2) = preprocess_aligned(
        adatas,
        key_class=key_class,
        df_varmap_1v1=df_varmap_1v1,
        node_source='deg,hvg',
        ntop_deg=50,
    )

    _ = main_for_aligned(
        **came_inputs,
        dataset_names=dsnames,
        key_class1=key_class,
        key_class2=key_class,
        do_normalize=True,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=100,
        params_model=dict(residual=False),
        batch_size=batch_size,
        plot_results=True,
    )

    del _
    torch.cuda.empty_cache()
    logging.debug('memory cleared\n')
    print('Test passed for ALIGNED!')


def __test2__(n_epochs: int = 5, batch_size=None, reverse=False):
    from .utils.train import seed_everything
    seed_everything()
    datadir = Path(os.path.abspath(__file__)).parent / 'sample_data'
    if not datadir.exists():
        from .utils._get_example_data import _extract_zip
        _extract_zip()

    sp1, sp2 = ('human', 'mouse')
    dsnames = ('Baron_human', 'Baron_mouse')
    if reverse:
        sp1, sp2 = sp2, sp1
        dsnames = dsnames[::-1]

    df_varmap_1v1 = pd.read_csv(datadir / f'gene_matches_1v1_{sp1}2{sp2}.csv', )
    df_varmap = pd.read_csv(datadir / f'gene_matches_{sp1}2{sp2}.csv', )

    dsn1, dsn2 = dsnames
    fp1, fp2 = datadir / f'raw-{dsn1}.h5ad', datadir / f'raw-{dsn2}.h5ad'
    if not fp1.exists():
        fp1 = datadir / f'raw-{dsn1}-sampled.h5ad'

    adata_raw1, adata_raw2 = sc.read_h5ad(fp1), sc.read_h5ad(fp2)
    adatas = [adata_raw1, adata_raw2]

    key_class = 'cell_ontology_class'
    time_tag = make_nowtime_tag()
    resdir = Path('_temp') / f'{dsnames}-{time_tag}'

    came_inputs, (adata1, adata2) = preprocess_unaligned(
        adatas,
        key_class=key_class,
        node_source='deg,hvg',
        ntop_deg=50,
    )

    _ = main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class,
        key_class2=key_class,
        do_normalize=True,
        keep_non1v1_feats=True,
        non1v1_trans_to=1,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=100,
        params_model=dict(residual=False),
        batch_size=batch_size,
        plot_results=True,
    )

    del _
    torch.cuda.empty_cache()
    logging.debug('memory cleared\n')
    print('Test passed for UN-ALIGNED!')
