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

from pathlib import Path
from typing import Any, Dict

from models.model_type import HiveModelType
from models.vanilla.base_model import BaseModel
from models.variational.vae_base_model import VaeBaseModel
from models.vanilla.contrastive.contrastive_base_model import ContrastiveBaseModel
from models.variational.contrastive.contrastive_variational_base_model import ContrastiveVariationalBaseModel
from models.variational.contrastive.contrastive_variational_base_model import latent_permutation

from typing import List, Callable, Union, Optional
from torch import nn, device

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from features.contrastive_feature_dataset import ContrastiveOutput
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
               output_path: Path, optimizer: Optimizer, epoch_no: int, loss: float) -> None:
    """
    Function for saving model on disc
    :param model: model to be saved
    :param output_path: filename with path for checkpoint file.
    :param optimizer: optimizer
    :param epoch_no: epoch number
    :param loss:
    """
    torch.save(
        {'epoch': epoch_no, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
         'loss': loss}, output_path)


def model_load(checkpoint_filepath: Path, model: Union[BM, VBM, CVBM, CBM, SmDataParallel],
               optimizer: Optimizer = None, gpu_ids=None):
    """
    Function for loading model from disc
    :param gpu_ids: ids of gpus to be used
    :param checkpoint_filepath: checkpoint path
    :param model: empty object of BaseClass
    :param optimizer: optimizer
    :return: epoch, loss
    """
    checkpoint = torch.load(checkpoint_filepath)
    if any([key.startswith('module') for key in checkpoint['model_state_dict'].keys()]):
        model = SmDataParallel(model, gpu_ids)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


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


def _prepare_discrimination_for_train(discriminator_model_config: dict, model_latent: int, torch_device: torch.device,
                                      discriminator_train_config: dict):
    """
    Method for preparing discriminator model and optimizer
    :param discriminator_model_config:
    :param model_latent:
    :param torch_device:
    :param discriminator_train_config:
    :return:
    """
    discriminator = HiveModelFactory.get_discriminator(discriminator_model_config, model_latent).to(torch_device)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch_device is not None:
            self.device = torch_device

        self.gpu_ids = gpu_ids if self.device == torch.device("cuda") else None
        self._curr_patience = -1
        self._curr_best_loss = sys.maxsize

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
                disc, disc_opt = _prepare_discrimination_for_train(optuna_model_config,
                                                                   optuna_model_config['model']['latent'],
                                                                   self.device,
                                                                   optuna_learning_config['discriminator'])

            for epoch in range(1, optuna_learning_config.get('epochs', 10) + 1):
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
            for (batch, _) in dataloader:
                output = torch.cat((output, model.get_latent(batch).cpu()))

        return output

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

    def train_contrastive(self,
                          model: CBM,
                          train_dataloader: DataLoader,
                          train_config: dict,
                          val_dataloader: DataLoader = None,
                          feature_config: dict = None) -> CBM:
        """
        Wrapper for training vanilla contrastive autoencoders (both convolutional and fully connected)
        :param feature_config: feature config to be logged
        :param val_dataloader: validation dataloader
        :param train_dataloader: train dataloader
        :param model: model to be trained
        :param train_config: train config
        """
        return self._train(model, train_dataloader, train_config, self._train_contrastive_step,
                           self._val_contrastive_step, val_dataloader, feature_config)

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
        if feature_config is None:
            feature_config = {}
        self._curr_best_loss = sys.maxsize
        self._curr_patience = train_config.get('epoch_patience', 10)
        patience_init_val = train_config.get('epoch_patience', 10)

        experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            {**model.get_params(),
                                             **train_config,
                                             **(feature_config if feature_config is not None else {})}, [])
        checkpoint_path = self.output_folder / f'{experiment.get_name()}-checkpoint.pth'

        log.debug(f'performing train task on {self.device}(s) ({torch.cuda.device_count()})'
                  f' for model {checkpoint_path.stem.upper()} with following config: {train_config}')

        model = SmDataParallel(model, self.gpu_ids) if torch.cuda.device_count() > 1 else model.to(self.device)

        optimizer: Optimizer = _parse_optimizer(train_config['model']['optimizer'].get('type', 'Adam'))(
            model.parameters(),
            lr=train_config.get(
                'learning_rate',
                0.0001))
        log_interval = train_config.get('logging_batch_interval', 10)
        for epoch in range(1, train_config.get('epochs', 10) + 1):
            train_epoch_loss = train_step_func(model, train_dataloader, optimizer, experiment, epoch, log_interval)
            experiment.log_metric('train_epoch_loss', train_epoch_loss.model_loss, step=epoch)

            log.info(f'--- train epoch {epoch} end with train loss: {train_epoch_loss.model_loss}')
            early_stopping_loss = train_epoch_loss
            if val_dataloader is not None:
                val_epoch_loss = val_step_func(model, val_dataloader, experiment, epoch, log_interval)
                experiment.log_metric('val_epoch_loss', val_epoch_loss.model_loss, step=epoch)
                early_stopping_loss = val_epoch_loss
                log.info(f'--- train epoch {epoch} end with val loss: {val_epoch_loss.model_loss} ---')

            patience = self.early_stopping_callback(early_stopping_loss.model_loss, patience_init_val)
            if patience == patience_init_val:
                log.debug(f'*** model checkpoint at epoch {epoch} ***')
                model_save(model, checkpoint_path, optimizer, epoch, train_epoch_loss.model_loss)
            elif patience == 0:
                log.info(f' ___ early stopping at epoch {epoch} ___')
                epoch, _ = model_load(checkpoint_path, model, optimizer, gpu_ids=self.gpu_ids)
                break

        return model

    def _train_contrastive_with_discriminator(self,
                                              model: CVBM,
                                              train_dataloader: DataLoader,
                                              train_config: dict,
                                              discriminator: nn.Module,
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
        self._curr_best_loss = sys.maxsize
        self._curr_patience = train_config.get('epoch_patience', 10)
        patience_init_val = train_config.get('epoch_patience', 10)

        experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            {**model.get_params(),
                                             **train_config,
                                             **(feature_config if feature_config is not None else {})}, [])

        model_checkpoint_path = self.output_folder / f'{experiment.get_name()}-contrastive-checkpoint.pth'
        discriminator_checkpoint_path = self.output_folder / f'{experiment.get_name()}-discriminator-checkpoint.pth'

        log.debug(f'performing train task on {self.device}(s) ({torch.cuda.device_count()})'
                  f' for model {model_checkpoint_path.stem.upper()} with following config: {train_config}')

        model, discriminator = (SmDataParallel(model, self.gpu_ids), SmDataParallel(discriminator, self.gpu_ids)) \
            if torch.cuda.device_count() > 1 else (model.to(self.device), discriminator.to(self.device))

        model_optimizer: Optimizer = _parse_optimizer(train_config['model']['optimizer'].get('type', 'Adam'))(
            model.parameters(), lr=train_config.get('learning_rate', 0.0001))
        discriminator_optimizer: Optimizer = _parse_optimizer(
            train_config['discriminator']['optimizer'].get('type', 'Adam'))(discriminator.parameters(),
                                                                            lr=train_config['discriminator'].get(
                                                                                'learning_rate', 0.0001))

        log_interval = train_config.get('logging_batch_interval', 10)
        for epoch in range(1, train_config.get('epochs', 10) + 1):
            epoch_loss = self._train_contrastive_step(model, train_dataloader,
                                                      model_optimizer, experiment, epoch, log_interval,
                                                      discriminator, discriminator_optimizer)
            experiment.log_metric('train_epoch_loss', epoch_loss.model_loss, step=epoch)
            experiment.log_metric('train_discriminator_epoch_loss', epoch_loss.discriminator_loss, step=epoch)

            log.info(f'--- train epoch {epoch} end with train loss: {epoch_loss.model_loss}')
            early_stopping_loss = epoch_loss
            if val_dataloader is not None:
                val_epoch_loss = self._val_contrastive_step(model, val_dataloader, experiment, epoch, log_interval)
                experiment.log_metric('val_epoch_loss', val_epoch_loss.model_loss, step=epoch)
                early_stopping_loss = val_epoch_loss
                log.info(f'--- train epoch {epoch} end with val loss: {val_epoch_loss.model_loss} ---')

            patience = self.early_stopping_callback(early_stopping_loss.model_loss, patience_init_val)
            if patience == patience_init_val:
                log.debug(f'*** model checkpoint at epoch {epoch} ***')
                model_save(model, model_checkpoint_path, model_optimizer, epoch, epoch_loss.model_loss)
                model_save(discriminator, discriminator_checkpoint_path, discriminator_optimizer, epoch,
                           epoch_loss.discriminator_loss)
            elif patience == 0:
                log.info(f' ___ early stopping at epoch {epoch} ___')
                epoch, _ = model_load(model_checkpoint_path, model, model_optimizer, gpu_ids=self.gpu_ids)
                break

        return model

    def _train_contrastive_step(self,
                                model: Union[CBM, CVBM, SmDataParallel],
                                dataloader: DataLoader,
                                model_optimizer: Optimizer,
                                experiment: Experiment,
                                epoch: int,
                                logging_interval: int,
                                discriminator: nn.Module = None,
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
        :param logging_interval: logging interval
        :return: epoch loss object
        """
        mean_loss = []
        discriminator_mean_loss = []
        model.train()
        for batch_idx, (target, background) in enumerate(dataloader):
            target_batch = target.to(self.device)
            background_batch = background.to(self.device)
            model_optimizer.zero_grad()
            model_output: ContrastiveOutput = model(target_batch, background_batch)
            loss = model.loss_fn(target_batch, background_batch, model_output, discriminator)
            loss.backward()
            model_optimizer.step()

            loss_float = loss.item()
            mean_loss.append(loss_float)

            experiment.log_metric("batch_train_loss", loss_float, step=epoch * batch_idx)
            if logging_interval != -1 and batch_idx % logging_interval == 0:
                log.info(f'=== train epoch {epoch},'
                         f' [{batch_idx * len(target)}/{len(dataloader.dataset)}] '
                         f'-> batch loss: {loss_float}')

            if all([discriminator, discriminator_optimizer]):
                q = torch.cat((model_output.target_qs_latent.clone().detach(),
                               model_output.target_qz_latent.clone().detach()), dim=-1).squeeze()
                q_bar = latent_permutation(q)
                q = q.to(self.device)
                q_bar = q_bar.to(self.device)
                discriminator_optimizer.zero_grad()
                q_score, q_bar_score = discriminator(q, q_bar)
                dloss = discriminator.loss_fn(q_score, q_bar_score)
                dloss.backward()
                discriminator_optimizer.step()

                discriminator_loss_float = dloss.item()
                discriminator_mean_loss.append(discriminator_loss_float)
                experiment.log_metric("discriminator_batch_train_loss", dloss, step=epoch * batch_idx)
                if logging_interval != -1 and batch_idx % logging_interval == 0:
                    log.info(f'-> discriminator loss: {discriminator_loss_float}')

        epoch_loss = EpochLoss(sum(mean_loss) / len(mean_loss))
        if len(discriminator_mean_loss) > 0:
            epoch_loss.discriminator_loss = sum(discriminator_mean_loss) / len(discriminator_mean_loss)
        return epoch_loss

    def _val_contrastive_step(self,
                              model: Union[CBM, CVBM, SmDataParallel],
                              val_dataloader: DataLoader,
                              experiment: Experiment,
                              epoch_no: int,
                              logging_interval: int) -> EpochLoss:
        """
        Function for performing validation step for model
        :param model: model to be evaluated
        :param val_dataloader: validation dataloader
        :param experiment: comet ml experiment
        :param epoch_no: epoch number for validation step - mostly for logging
        :param logging_interval: interval for logs within epoch
        :return:
        """
        val_loss = []
        model.eval()
        with torch.no_grad:
            for batch_idx, (target, background) in enumerate(val_dataloader):
                target_batch = target.to(self.device)
                background_batch = background.to(self.device)
                model_output = model(target_batch, background_batch)

                loss = model.loss_fn(target_batch, background_batch, model_output)
                loss_float = loss.item()
                val_loss.append(loss_float)
                experiment.log_metric("batch_val_loss", loss_float, step=epoch_no * batch_idx)

                if logging_interval != -1 and batch_idx % logging_interval == 0:
                    log.info(f'=== validation epoch {epoch_no}, '
                             f'[{batch_idx * len(target)}/{len(val_dataloader.dataset)}]'
                             f'-> batch loss: {loss.item()} ===')

        return EpochLoss(sum(val_loss) / len(val_loss))

    T_train_step = Callable[[Union[BM, CBM, SmDataParallel], DataLoader, Optimizer,
                             Experiment, int, int], EpochLoss]

    def _train_step(self,
                    model: Union[BM, VBM, CBM, SmDataParallel],
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    experiment: Experiment,
                    epoch_no: int,
                    logging_interval: int = 10) -> EpochLoss:
        """
        Function for performing epoch step on data
        :param model: model to be trained
        :param dataloader: train dataloader
        :param optimizer: used optimizer
        :param experiment: comet ml experiment where data will be reported
        :param epoch_no: epoch number
        :param logging_interval: after how many batches data will be logged
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
            experiment.log_metric("batch_train_loss", loss_float, step=epoch_no * batch_idx)

            if logging_interval != -1 and batch_idx % logging_interval == 0:
                log.info(f'=== train epoch {epoch_no},'
                         f' [{batch_idx * len(batch)}/{len(dataloader.dataset)}] '
                         f'-> batch loss: {loss_float}')
        return EpochLoss(sum(mean_loss) / len(mean_loss))

    T_val_step = Callable[[Union[BM, CBM, CVBM, SmDataParallel], DataLoader,
                           Experiment, int, int], EpochLoss]

    def _val_step(self,
                  model: Union[BM, VBM, CBM, SmDataParallel],
                  val_dataloader: DataLoader,
                  experiment: Experiment,
                  epoch_no: int,
                  logging_interval: int) -> EpochLoss:
        """
        Function for performing validation step for model
        :param model: model to be evaluated
        :param val_dataloader: validation dataloader
        :param experiment: comet ml experiment
        :param epoch_no: epoch number for validation step - mostly for logging
        :param logging_interval: interval for logs within epoch
        :return:
        """
        val_loss = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (batch, _) in enumerate(val_dataloader):
                batch = batch.to(self.device)
                model_output = model(batch)
                loss = model.loss_fn(batch, model_output)

                loss_float = loss.item()
                val_loss.append(loss_float)
                experiment.log_metric("batch_val_loss", loss_float, step=epoch_no * batch_idx)

                if logging_interval != -1 and batch_idx % logging_interval == 0:
                    log.info(f'=== validation epoch {epoch_no}, '
                             f'[{batch_idx * len(batch)}/{len(val_dataloader.dataset)}]'
                             f'-> batch loss: {loss.item()} ===')

        return EpochLoss(sum(val_loss) / len(val_loss))

    def early_stopping_callback(self, curr_loss: float, patience_reset_value: int) -> int:
        """
        Method for performing callback for early stopping
        :param curr_loss:
        :param patience_reset_value:
        :return:
        """
        if self._curr_patience == 0 or math.isnan(curr_loss):
            return 0
        elif curr_loss < self._curr_best_loss:
            self._curr_best_loss = curr_loss
            self._curr_patience = patience_reset_value
        else:
            self._curr_patience -= 1

        return self._curr_patience
