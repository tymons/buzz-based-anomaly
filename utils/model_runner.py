from __future__ import annotations

import os
import sys
import logging

from comet_ml import Experiment

import torch
import time
import optuna
import math
import gc
import utils.utils as util

from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict

from models.discriminator import Discriminator
from models.model_type import HiveModelType
from models.vanilla.base_model import BaseModel
from models.variational.vae_base_model import VaeBaseModel
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel, VanillaContrastiveOutput
from models.variational.contrastive.contrastive_variational_base_model import ContrastiveVariationalBaseModel
from models.variational.contrastive.contrastive_variational_base_model import VaeContrastiveOutput

from typing import List, Callable, Union, Optional
from torch import nn, device

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from features.contrastive_feature_dataset import ContrastiveFeatureLogger
from utils.model_factory import HiveModelFactory, build_optuna_model_config
from utils.sm_data_parallel import SmDataParallel

CVBM = ContrastiveVariationalBaseModel
CBM = ContrastiveBaseModel
VBM = VaeBaseModel
BM = BaseModel


@dataclass
class EpochLoss:
    model_loss: float
    discriminator_loss: Optional[float] = None
    tc_loss: Optional[float] = None
    recon_loss: Optional[float] = None


log = logging.getLogger("smartula")


def clear_memory(model: Union[BM, VBM, CBM, SmDataParallel],
                 optimizer: torch.optim.Optimizer,
                 discriminator: nn.Module = None,
                 discriminator_optimizer: torch.optim.Optimizer = None):
    """
    Method for clearing all the memory from models and optimizers
    :param model:
    :param optimizer:
    :param discriminator:
    :param discriminator_optimizer:
    """
    transfer_optimizer_to(optimizer, torch.device('cpu'))
    model.to(torch.device('cpu'))
    del model
    del optimizer

    if all([discriminator, discriminator_optimizer]):
        transfer_optimizer_to(discriminator_optimizer, torch.device('cpu'))
        discriminator.to(torch.device('cpu'))
        del discriminator

    gc.collect()
    torch.cuda.empty_cache()


def _parse_optimizer(optimizer_name: str) -> Callable:
    """
    Function for getting parsing based on string representation
    :param optimizer_name:  string name representation
    :return: Optimizer
    """
    optimizer: dict = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
    }
    return optimizer.get(optimizer_name, lambda x: log.error(f'loss function for model {x} not implemented!'))


def transfer_optimizer_to(optimizer: Optimizer, device_to: device) -> Optimizer:
    """
    Closure for optimizier transfers
    :param optimizer: optimizer to be transferred
    :param device_to: torch device (cpu or gpu)
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device_to)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device_to)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device_to)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device_to)

    return optimizer


def _read_comet_key(path: Path) -> str:
    """
    Function for reading comet key from config file
    :param path: path to comet config file
    :return:
    """
    with path.open('r') as f:
        return [b.split('=')[-1] for b in f.read().splitlines() if b.startswith('api_key')][0]


def model_save(model: Union[BM, VBM, CBM, CVBM, SmDataParallel],
               output_path: Path, optimizer: Optimizer, epoch_no: int) -> None:
    """
    Function for saving model on disc
    :param model: model to be saved
    :param output_path: filename with path for checkpoint file.
    :param optimizer: optimizer
    :param epoch_no: epoch number
    :param loss:
    """
    torch.save(
        {'epoch': epoch_no, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
        output_path)


def model_load(checkpoint_filepath: Path, model: Union[BM, VBM, CVBM, CBM, SmDataParallel],
               optimizer: Optimizer = None, gpu_ids=None):
    """
    Function for loading model from disc
    :param gpu_ids: gpus ids
    :param checkpoint_filepath: checkpoint path
    :param model: empty object of BaseClass
    :param optimizer: optimizer
    :return: epoch, loss
    """
    checkpoint = torch.load(checkpoint_filepath)
    if any([key.startswith('model.module') for key in checkpoint['model_state_dict'].keys()]):
        model = SmDataParallel(model, gpu_ids)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, epoch


def modify_optuna_learning_config(learning_config: dict, trial: optuna.Trial) -> Dict[str, Any]:
    """
    Function for building optuna learning config
    :param learning_config: dictionary with learning config
    :param trial: optuna trial
    :return: directory with combined optuna suggest values and original learning dict
    """
    optuna_learning_config = learning_config.copy()
    optuna_learning_config['model']['learning_rate'] = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    optuna_learning_config['model']['optimizer']['type'] = trial.suggest_categorical('optimizer_type',
                                                                                     ['Adam', 'SGD', 'RMSprop',
                                                                                      'Adagrad', 'Adadelta'])
    if optuna_learning_config.get('discriminator') is not None:
        optuna_learning_config['discriminator']['optimizer']['type'] = trial.suggest_categorical('disc_optimizer_type',
                                                                                                 ['Adam', 'SGD',
                                                                                                  'RMSprop',
                                                                                                  'Adagrad',
                                                                                                  'Adadelta'])
        optuna_learning_config['discriminator']['learning_rate'] = trial.suggest_loguniform('discriminator_lr',
                                                                                            1e-5, 1e-2)
    return optuna_learning_config


def _prepare_discrimination_for_train(model_latent: int, torch_device: torch.device,
                                      discriminator_train_config: dict):
    """
    Method for preparing discriminator model and optimizer
    :param model_latent:
    :param torch_device:
    :param discriminator_train_config:
    :return:
    """
    discriminator = HiveModelFactory.get_discriminator(model_latent).to(torch_device)
    discriminator_optimizer_class = _parse_optimizer(discriminator_train_config['optimizer']['type'])
    discriminator_optimizer = discriminator_optimizer_class(discriminator.parameters(),
                                                            lr=discriminator_train_config['learning_rate'])
    return discriminator, discriminator_optimizer


class ModelRunner:
    device: device

    def __init__(self, output_folder: Path = None, comet_api_key: str = None, comet_config_file: Path = None,
                 comet_project_name: str = "Default Project", torch_device: torch.device = None,
                 gpu_ids: List[int] = None):
        self.comet_api_key = comet_api_key if comet_api_key is not None else _read_comet_key(comet_config_file)
        self.comet_proj_name = comet_project_name

        self.output_folder = output_folder
        if output_folder is not None:
            self.output_folder.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch_device is not None:
            self.device = torch_device

        self.gpu_ids = gpu_ids if self.device == torch.device("cuda") else None
        self._curr_patience = -1
        self._curr_best_loss = sys.maxsize
        self._patience_init_val = 10
        self._save_last_model_flag = False

    def _setup_experiment(self, experiment_name: str, log_parameters: dict, tags: List[str]) -> Experiment:
        """
        Method for setting up the comet ml experiment
        :param experiment_name: unique experiment name
        :param log_parameters: additional parameters for COMET ML experiment to be logged
        :param tags: additional tags for COMET ML experiment
        :return: Experiment
        """
        environment = os.getenv('ENVIRONMENT', None)
        if environment is not None:
            tags.append(environment)

        experiment = Experiment(api_key=self.comet_api_key, display_summary_level=0,
                                project_name=self.comet_proj_name, auto_metric_logging=False)
        experiment.set_name(experiment_name)
        experiment.log_parameters(log_parameters)
        experiment.add_tags(tags)
        return experiment

    def find_best(self, model_type: HiveModelType, train_dataloader: DataLoader, learning_config: dict, n_trials=10,
                  output_folder: Path = Path(__file__).parents[1].absolute(), feature_config: dict = None) -> None:
        """
        Method for searching best architecture with oputa
        :param model_type: autoencoder model type
        :param train_dataloader: train dataloader
        :param learning_config: train config
        :param n_trials: how many trials should be performed for optuna search
        :param output_folder: folder where best architecture config will be saved
        :param feature_config: feature config
        """
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
        study.optimize(lambda op_trial: self._optuna_train_objective(
            op_trial, model_type, train_dataloader, learning_config, feature_config), n_trials=n_trials)

        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        best_architecture_file = Path(output_folder) / Path(f"{model_type.model_name.lower()}"
                                                            f"-{time.strftime('%Y%m%d-%H%M%S')}.config")
        log.info('Study statistics: ')
        log.info(f'  Number of finished trials: {len(study.trials)}')
        log.info(f'  Number of pruned trials: {len(pruned_trials)}')
        log.info(f'  Number of complete trials: {len(complete_trials)}')

        log.info("Best trial:")
        trial = study.best_trial

        with best_architecture_file.open('w+') as f:
            f.write(f'Best loss: {str(trial.value)} \r\n')
            f.write('Params: \r\n')
            log.info(f'  Value: {trial.value}')
            log.info('  Params: ')
            for key, value in trial.params.items():
                log.info(f'    {key}:{value}')
                f.write(f'    {key}:{value} \r\n')

    def _optuna_train_objective(self,
                                trial: optuna.Trial,
                                model_type: HiveModelType,
                                train_dataloader: DataLoader,
                                learning_config: dict,
                                feature_config: dict = None) -> float:
        """
        Method for optuna objective
        :param trial: optuna.trial.Trial object
        :param model_type: model type which should be build
        :param train_dataloader: dataloader for train method
        :param learning_config: configuration for model trianing
        :param feature_config: feature config to be logged
        :return: final loss
        """
        optuna_model_config = build_optuna_model_config(model_type,
                                                        train_dataloader.dataset[0][0].squeeze().shape, trial)
        optuna_learning_config: Dict[str, Any] = modify_optuna_learning_config(learning_config, trial)

        self._curr_best_loss = sys.maxsize
        self._curr_patience = optuna_learning_config.get('epoch_patience', 10)
        try:
            model = HiveModelFactory.build_model(model_type, train_dataloader.dataset[0][0].squeeze().shape,
                                                 optuna_model_config['model'])
            experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                                {**model.get_params(),
                                                 **optuna_learning_config,
                                                 **(feature_config if feature_config is not None else {})},
                                                ['optuna'])

            log.debug(f'performing optuna train task on {self.device}(s) ({torch.cuda.device_count()})'
                      f' for model {type(model).__name__.lower()} with following config: {optuna_learning_config}')

            if torch.cuda.device_count() > 1:
                model = SmDataParallel.SmDataParallel(model, device_ids=self.gpu_ids)
            model = model.to(self.device)

            optimizer_class = _parse_optimizer(optuna_learning_config['model']['optimizer']['type'])
            optim = optimizer_class(model.parameters(), lr=optuna_learning_config['model']['learning_rate'])
            log_interval = optuna_learning_config.get('logging_batch_interval', 10)

            disc, disc_opt = None, None
            if model_type.num >= HiveModelType.CONTRASTIVE_VAE.num:
                disc, disc_opt = _prepare_discrimination_for_train(optuna_model_config['model']['latent'],
                                                                   self.device, optuna_learning_config['discriminator'])

            for epoch in range(optuna_learning_config.get('epochs', 10)):
                epoch_loss = self._train_step(model, train_dataloader, optim, experiment, epoch, log_interval) \
                    if model_type.num < HiveModelType.CONTRASTIVE_AE.num else \
                    self._train_contrastive_step(model, train_dataloader,
                                                 optim, experiment, epoch, log_interval, disc, disc_opt)

                epoch_loss_value = epoch_loss.model_loss
                experiment.log_metric('train_epoch_loss', epoch_loss_value, step=epoch)
                log.info(f'--- train epoch {epoch} end with train loss: {epoch_loss_value} ---')

                trial.report(epoch_loss_value, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                self.early_stopping_callback(epoch_loss_value, optuna_learning_config.get('epochs', 10) + 1)

            # clear memory
            clear_memory(model, optim)

            return self._curr_best_loss
        except RuntimeError as e:
            log.error(f'hive model build failed for config: {optuna_model_config} with exception: {e}')

    def inference_latent(self, model: Union[BM, VBM, CBM, CVBM], dataloader: DataLoader):
        """
        Wrapper for model inference
        :param dataloader: data where inference should be performed
        :param model: model
        :return:
        """
        model = model.to(self.device)
        model.eval()
        output = torch.Tensor()
        with torch.no_grad():
            for (batch, _) in tqdm(dataloader):
                batch = batch.to(self.device)
                output = torch.cat((output, model.get_latent(batch).cpu()))

        return output.squeeze()

    def train(self,
              model: Union[BM, VBM],
              train_dataloader: DataLoader,
              train_config: dict,
              val_dataloader: DataLoader = None,
              feature_config: dict = None) -> BM:
        """
        Wrapper for training vanilla or variational autoencoders (both convolutional and fully connected)
        :param feature_config: feature config to be logged
        :param val_dataloader: validation dataloader
        :param train_dataloader: train dataloader
        :param model: model to be trained
        :param train_config: train config
        :return:
        """
        return self._train(model, train_dataloader, train_config, self._train_step, self._val_step,
                           val_dataloader, feature_config)

    def train_contrastive_with_discriminator(self,
                                             model: CVBM,
                                             train_dataloader: DataLoader,
                                             train_config: dict,
                                             discriminator: nn.Module,
                                             val_dataloader: DataLoader = None,
                                             feature_config: dict = None) -> CVBM:
        """
        Wrapper for training contrastive variational autoencoders
        :param model: contrastive model to be trained
        :param train_dataloader: train dataloader object
        :param train_config: learning config dictionary
        :param discriminator: discriminator to be used for latent class learning
        :param val_dataloader: validation dataloader
        :param feature_config: dict for feature config to be logged with comet ml
        :return:
        """
        return self._train_contrastive_with_discriminator(model, train_dataloader, train_config, discriminator,
                                                          val_dataloader, feature_config)

    def _train(self, model: Union[BM, VBM, CBM],
               train_dataloader: DataLoader,
               train_config: dict,
               train_step_func: T_train_step,
               val_step_func: T_val_step,
               val_dataloader: DataLoader = None,
               feature_config=None) -> (Union[BM, CBM]):
        """
        Base function for training model with specified config.
        This method should be used only without discriminator model.
        :param model: model to be trained
        :param train_config: configuration for learning process
        :param train_step_func: function to be invoked for single epoch training
        :param val_step_func: function to be invoked for single epoch validating
        :rtype: BM: trained model, last loss
        """
        experiment, checkpoint_folder, run_folder = self._initialize_training(train_config, feature_config, model)
        checkpoint_path = checkpoint_folder / f'{experiment.get_name()}-checkpoint.pth'

        model = SmDataParallel(model, self.gpu_ids) if torch.cuda.device_count() > 1 else model.to(self.device)

        optimizer: Optimizer = _parse_optimizer(train_config['model']['optimizer'].get('type', 'Adam'))(
            model.parameters(), lr=train_config.get('learning_rate', 0.0001))

        _ = val_step_func(model, val_dataloader, -1, None, run_folder)

        log.debug(f'performing train task on {self.device}(s) ({torch.cuda.device_count()})'
                  f' for model {checkpoint_path.stem.upper()} with following config: {train_config}')
        for epoch in range(train_config.get('epochs', 10)):
            # ------------ train step ------------
            train_epoch_loss = train_step_func(model, train_dataloader, optimizer, experiment, epoch)
            experiment.log_metric('train_epoch_loss', train_epoch_loss.model_loss, step=epoch)
            log.info(f'--- train epoch {epoch} end with train loss: {train_epoch_loss.model_loss}')

            # ------------ validation step ------------
            val_epoch_loss = val_step_func(model, val_dataloader, epoch, experiment, run_folder)
            experiment.log_metric('val_epoch_loss', val_epoch_loss.model_loss, step=epoch)
            log.info(f'--- validation epoch {epoch} end with val loss: {val_epoch_loss.model_loss} ---')

            # ------------ early stopping handler ------------
            should_stop = self._early_stopping_handler(val_epoch_loss.model_loss,
                                                       epoch, model, optimizer, checkpoint_path)
            if should_stop:
                log.info(f' ___ early stopping at epoch {epoch} ___')
                break

            if self._save_last_model_flag is True:
                model_save(model, checkpoint_folder / f'{experiment.get_name()}-last-epoch.pth', optimizer, epoch)

        return model

    def _train_contrastive_with_discriminator(self,
                                              model: CVBM,
                                              train_dataloader: DataLoader,
                                              train_config: dict,
                                              discriminator: Discriminator,
                                              val_dataloader: DataLoader = None,
                                              feature_config: dict = None) -> (CVBM, float):
        """
        Method for training contrastive model along with discriminator. Most often this method should be used for
        variational contrastive autoencoders
        :param model:
        :param train_config:
        :param discriminator:
        :return trained model, last loss
        """
        experiment, checkpoint_folder, run_folder = self._initialize_training(train_config, feature_config, model)
        model_checkpoint_path = checkpoint_folder / f'{experiment.get_name()}-contrastive-checkpoint.pth'
        disc_checkpoint_path = checkpoint_folder / f'{experiment.get_name()}-discriminator-checkpoint.pth'

        model, discriminator = (SmDataParallel(model, self.gpu_ids), SmDataParallel(discriminator, self.gpu_ids)) \
            if torch.cuda.device_count() > 1 else (model.to(self.device), discriminator.to(self.device))

        model_optimizer: Optimizer = _parse_optimizer(train_config['model']['optimizer'].get('type', 'Adam'))(
            model.parameters(), lr=train_config.get('learning_rate', 0.0001))
        discriminator_optimizer: Optimizer = _parse_optimizer(
            train_config['discriminator']['optimizer'].get('type', 'Adam'))(discriminator.parameters(),
                                                                            lr=train_config['discriminator'].get(
                                                                                'learning_rate', 0.0001))

        _ = self._val_contrastive_step(model, val_dataloader, -1, experiment=None, discriminator=discriminator,
                                       fig_folder=run_folder)

        log.debug(f'running train task on {self.device}(s) ({torch.cuda.device_count()}) for model '
                  f'{model_checkpoint_path.stem.upper()} with following config: {train_config}')
        for epoch in range(train_config.get('epochs', 10)):
            # ------------ train step ------------
            epoch_loss = self._train_contrastive_step(model, train_dataloader,
                                                      model_optimizer, experiment, epoch, discriminator,
                                                      discriminator_optimizer)
            experiment.log_metric('train_epoch_loss', epoch_loss.model_loss, step=epoch)
            experiment.log_metric('train_discriminator_epoch_loss', epoch_loss.discriminator_loss, step=epoch)
            experiment.log_metric('train_tc_epoch_loss', epoch_loss.tc_loss, step=epoch)
            log.info(f'--- train epoch {epoch} end with train loss: {epoch_loss.model_loss}')

            # ------------ validation step ------------
            val_epoch_loss = self._val_contrastive_step(model, val_dataloader, epoch, experiment=experiment,
                                                        discriminator=discriminator, fig_folder=run_folder)
            experiment.log_metric('val_epoch_loss', val_epoch_loss.model_loss, step=epoch)
            log.info(f'--- validation epoch {epoch} end with val loss: {val_epoch_loss.model_loss} ---')

            # ------------ early stopping handler ------------
            es_loss = val_epoch_loss.recon_loss + epoch_loss.tc_loss + epoch_loss.discriminator_loss
            should_stop = self._early_stopping_handler(es_loss, epoch, model, model_optimizer,
                                                       model_checkpoint_path, discriminator, discriminator_optimizer,
                                                       disc_checkpoint_path)
            if should_stop:
                log.info(f' ___ early stopping at epoch {epoch} ___')
                break

            if self._save_last_model_flag is True:
                model_save(model, model_checkpoint_path / f'{experiment.get_name()}-last-epoch.pth', model_optimizer,
                           epoch)

        return model

    def _train_contrastive_step(self,
                                model: Union[CBM, CVBM, SmDataParallel],
                                dataloader: DataLoader,
                                model_optimizer: Optimizer,
                                experiment: Experiment,
                                epoch: int,
                                discriminator: Discriminator = None,
                                discriminator_optimizer: Optimizer = None) -> EpochLoss:
        """
        Function for epoch run on contrastive model
        :param model: model to be train on contrastive manner
        :param dataloader: train data loader
        :param model_optimizer: optimizer for contrastive model
        :param discriminator: latent discriminator model
        :param discriminator_optimizer: optimizer for latent discriminator model
        :param experiment: comet ml experiment
        :param epoch: current epoch
        :return: epoch loss object
        """
        mean_loss = []
        discriminator_mean_loss = []
        tc_mean_loss = []
        recon_mean_loss = []

        model.train()
        for batch_idx, (target, background) in enumerate(dataloader):
            target_batch = target.to(self.device)
            background_batch = background.to(self.device)
            model_optimizer.zero_grad()
            model_output: Union[VaeContrastiveOutput, VanillaContrastiveOutput] = model(target_batch, background_batch)
            loss, partial_loss = model.loss_fn(target_batch, background_batch, model_output, discriminator)
            recon_loss, disc_loss, tc_loss = partial_loss

            loss.backward()
            model_optimizer.step()
            loss_float = loss.item()
            mean_loss.append(loss_float)

            experiment.log_metric("batch_train_loss", loss_float, step=(epoch * len(dataloader)) + batch_idx)
            experiment.log_metric("batch_recon_loss", recon_loss, step=(epoch * len(dataloader)) + batch_idx)
            experiment.log_metric("batch_tc_loss", tc_loss, step=(epoch * len(dataloader)) + batch_idx)
            tc_mean_loss.append(tc_loss)
            recon_mean_loss.append(recon_loss)
            discriminator_mean_loss.append(disc_loss)

            if self._log_interval != -1 and batch_idx % self._log_interval == 0:
                log.info(f'=== train epoch {epoch}, [{batch_idx * len(target)}/{len(dataloader.dataset)}] '
                         f'-> batch loss: {loss_float}')

            if all([discriminator, discriminator_optimizer]):
                target_latent = model_output.target_latent.squeeze(dim=1).clone().detach()
                background_latent = model_output.background_latent.squeeze(dim=1).clone().detach()

                latent_data = torch.vstack((target_latent, background_latent)).to(self.device)
                latent_labels = torch.hstack((torch.ones(target_latent.shape[0]),
                                              torch.zeros(background_latent.shape[0]))).reshape(-1, 1).to(self.device)

                scores = discriminator(latent_data)
                dloss = discriminator.loss_fn(scores, latent_labels)
                dloss.backward()
                discriminator_optimizer.step()
                discriminator_loss_float = dloss.item()

                if self._log_interval != -1 and batch_idx % self._log_interval == 0:
                    log.info(f'-> discriminator loss: {discriminator_loss_float}')
                experiment.log_metric("batch_disc_loss", discriminator_loss_float,
                                      step=(epoch * len(dataloader)) + batch_idx)

        epoch_loss = EpochLoss(sum(mean_loss) / len(mean_loss),
                               discriminator_loss=sum(discriminator_mean_loss) / len(discriminator_mean_loss),
                               tc_loss=sum(tc_mean_loss) / len(tc_mean_loss),
                               recon_loss=sum(recon_mean_loss) / len(recon_mean_loss))

        return epoch_loss

    def _val_contrastive_step(self,
                              model: Union[CBM, CVBM, SmDataParallel],
                              val_dataloader: DataLoader,
                              epoch_no: int,
                              experiment: Experiment = None,
                              discriminator: Discriminator = None,
                              fig_folder: Path = None) -> EpochLoss:
        """
        Function for performing validation step for model
        :param model: model to be evaluated
        :param val_dataloader: validation dataloader
        :param experiment: comet ml experiment
        :param epoch_no: epoch number for validation step - mostly for logging
        :param discriminator: discriminator to be used for vaes and density ratio trick
        :return:
        """
        val_loss = []
        model.eval()
        contrastive_logger = ContrastiveFeatureLogger(fig_folder, type(model)) if fig_folder is not None else None

        with torch.no_grad():
            for batch_idx, (target, background) in enumerate(val_dataloader):
                target_batch = target.to(self.device)
                background_batch = background.to(self.device)
                model_output: Union[VaeContrastiveOutput, VanillaContrastiveOutput] = model(target_batch,
                                                                                            background_batch)

                loss, partial_losses = model.loss_fn(target_batch, background_batch, model_output, discriminator)
                recon_loss, disc_loss, tc_loss = partial_losses
                loss_float = loss.item()

                val_loss.append(loss_float)
                if experiment is not None:
                    experiment.log_metric("batch_val_loss", loss_float,
                                          step=(epoch_no * len(val_dataloader)) + batch_idx)

                if self._log_interval != -1 and batch_idx % self._log_interval == 0:
                    log.info(f'=== validation epoch {epoch_no}, '
                             f'[{batch_idx * len(target)}/{len(val_dataloader.dataset)}]'
                             f'-> batch loss: {loss.item()} ===')

                if contrastive_logger is not None:
                    contrastive_logger.batch_collect(model_output)

        if contrastive_logger is not None:
            contrastive_logger.data_flush(epoch_no)

        return EpochLoss(sum(val_loss) / len(val_loss), discriminator_loss=disc_loss,
                         tc_loss=tc_loss, recon_loss=recon_loss)

    T_train_step = Callable[[Union[BM, CBM, SmDataParallel], DataLoader, Optimizer,
                             Experiment, int], EpochLoss]

    def _train_step(self,
                    model: Union[BM, VBM, CBM, SmDataParallel],
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    experiment: Experiment,
                    epoch_no: int, ) -> EpochLoss:
        """
        Function for performing epoch step on data
        :param model: model to be trained
        :param dataloader: train dataloader
        :param optimizer: used optimizer
        :param experiment: comet ml experiment where data will be reported
        :param epoch_no: epoch number
        :return: mean loss for all batches
        """
        mean_loss = []
        model.train()
        for batch_idx, (batch, _) in enumerate(dataloader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            model_output = model(batch)
            loss = model.loss_fn(batch, model_output)
            loss.backward()
            optimizer.step()

            loss_float = loss.item()
            mean_loss.append(loss_float)
            experiment.log_metric("batch_train_loss", loss_float, step=(epoch_no * len(dataloader)) + batch_idx)

            if self._log_interval != -1 and batch_idx % self._log_interval == 0:
                log.info(f'=== train epoch {epoch_no},'
                         f' [{batch_idx * len(batch)}/{len(dataloader.dataset)}] '
                         f'-> batch loss: {loss_float}')
        return EpochLoss(sum(mean_loss) / len(mean_loss))

    T_val_step = Callable[[Union[BM, CBM, CVBM, SmDataParallel], DataLoader,
                           int, Optional[Experiment], Optional[Path]], EpochLoss]

    def _val_step(self,
                  model: Union[BM, VBM, CBM, SmDataParallel],
                  val_dataloader: DataLoader,
                  epoch_no: int,
                  experiment: Experiment = None,
                  fig_folder: Optional[Path] = None) -> EpochLoss:
        """
        Function for performing validation step for model
        :param model: model to be evaluated
        :param val_dataloader: validation dataloader
        :param experiment: comet ml experiment
        :param epoch_no: epoch number for validation step - mostly for logging
        :return:
        """
        val_loss = []
        model.eval()
        cat_latent = torch.Tensor() if fig_folder is not None else None

        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(val_dataloader):
                batch = batch.to(self.device)
                model_output = model(batch)
                loss = model.loss_fn(batch, model_output)

                loss_float = loss.item()
                val_loss.append(loss_float)
                if experiment is not None:
                    experiment.log_metric("batch_val_loss", loss_float,
                                          step=(epoch_no * len(val_dataloader)) + batch_idx)

                if self._log_interval != -1 and batch_idx % self._log_interval == 0:
                    log.info(f'=== validation epoch {epoch_no}, '
                             f'[{batch_idx * len(batch)}/{len(val_dataloader.dataset)}]'
                             f'-> batch loss: {loss.item()} ===')

                if cat_latent is not None:
                    cat_latent = torch.cat((cat_latent, model.get_latent(batch).cpu()), dim=0)

        if fig_folder is not None:
            util.plot_latent(cat_latent, fig_folder, epoch_no, experiment=experiment)

        return EpochLoss(sum(val_loss) / len(val_loss))

    def early_stopping_callback(self, curr_loss: float) -> int:
        """
        Method for performing callback for early stopping
        :param curr_loss:
        :return:
        """
        if self._curr_patience == 0 or math.isnan(curr_loss):
            return 0
        elif curr_loss < self._curr_best_loss:
            self._curr_best_loss = curr_loss
            self._curr_patience = self._patience_init_val
        else:
            self._curr_patience -= 1

        return self._curr_patience

    def _early_stopping_handler(self, loss, current_epoch, model, model_optimizer, model_checkpoint_path,
                                discriminator=None, discriminator_optimizer=None, discriminator_checkpoint_path=None):
        patience = self.early_stopping_callback(loss)
        if patience == self._patience_init_val:
            log.debug(f'*** model checkpoint at epoch {current_epoch} ***')
            model_save(model, model_checkpoint_path, model_optimizer, current_epoch)
            if all([discriminator, discriminator_checkpoint_path, discriminator_optimizer]):
                model_save(discriminator, discriminator_checkpoint_path, discriminator_optimizer, current_epoch)
        elif patience == 0:
            model, _ = model_load(model_checkpoint_path, model, model_optimizer, gpu_ids=self.gpu_ids)
            return True

        return False

    def _initialize_training(self, train_config, feature_config, model):
        """
        Method for initializing train run
        :param train_config: train config
        :param feature_config: feature config
        :param model: model to be trained
        :return: comet_ml experiment, folder for model checkpoints, folder for debug plotting
        """
        self._curr_best_loss = sys.maxsize
        self._curr_patience = train_config.get('epoch_patience', 10)
        self._patience_init_val = train_config.get('epoch_patience', 10)
        self._save_last_model_flag = train_config.get('save_last', False)
        self._log_interval = train_config.get('logging_batch_interval', 10)

        experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            {**model.get_params(), **train_config,
                                             **(feature_config if feature_config is not None else {})}, [])

        model_debug_folder = self.output_folder / f'runs/{experiment.get_name()}'
        model_checkpoint_folder = self.output_folder / f'models/{experiment.get_name()}'
        model_debug_folder.mkdir(exist_ok=True, parents=True)
        model_checkpoint_folder.mkdir(exist_ok=True, parents=True)

        return experiment, model_checkpoint_folder, model_debug_folder
