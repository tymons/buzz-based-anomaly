import sys

from comet_ml import Experiment

import torch
import time
import optuna
import math

from pathlib import Path

from models.base_model import BaseModel
from typing import List, Callable, Union
from torch import nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from models.model_type import HiveModelType
from utils.model_factory import HiveModelFactory, build_optuna_ae_config


def _read_comet_key(path: Path) -> str:
    """
    Function for reading comet key from config file
    :param path: path to comet config file
    :return:
    """
    with path.open('r') as f:
        return [b.split('=')[-1] for b in f.read().splitlines() if b.startswith('api_key')][0]


def _parse_optimizer(optimizer_name: str) -> Callable:
    """
    Function for getting optimizer based on string representation
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
    return optimizer.get(optimizer_name, lambda x: print(f'loss function for model {x} not implemented!'))


def model_save(model: BaseModel, output_path: Path, optimizer: Optimizer, epoch_no: int,
               loss: float) -> None:
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


def model_load(checkpoint_filepath: Path, model: BaseModel, optimizer: Optimizer = None):
    """
    Function for loading model from disc
    :param checkpoint_filepath: checkpoint path
    :param model: empty object of BaseClass
    :param optimizer: optimizer
    :return: epoch, loss
    """
    checkpoint = torch.load(checkpoint_filepath)
    if any([key.startswith('module') for key in checkpoint['model_state_dict'].keys()]):
        model = nn.DataParallel(model)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


class ModelRunner:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, output_folder: Path,
                 feature_config=None, comet_api_key: str = None, comet_config_file: Path = None,
                 comet_project_name: str = "Default Project"):
        if feature_config is None:
            feature_config = {}
        self.comet_api_key = comet_api_key if comet_api_key is not None else _read_comet_key(comet_config_file)
        self.comet_proj_name = comet_project_name
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.output_folder = output_folder
        self.feature_config = feature_config if feature_config is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_experiment(self, experiment_name: str, log_parameters: dict, tags: List[str]) -> Experiment:
        """
        Method for setting up the comet ml experiment
        :param experiment_name: unique experiment name
        :param log_parameters: additional parameters for COMET ML experiment to be logged
        :param tags: additional tags for COMET ML experiment
        :return: Experiment
        """
        experiment = Experiment(api_key=self.comet_api_key,
                                project_name=self.comet_proj_name, auto_metric_logging=False)
        experiment.set_name(experiment_name)
        experiment.log_parameters(log_parameters)
        experiment.add_tags(tags)
        return experiment

    def _train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                    experiment: Experiment, epoch_no: int, logging_interval: int = 10) -> float:
        """
        Function for performing epoch step on data
        :param model: model to be trained
        :param optimizer: used optimizer
        :param experiment: comet ml experiment where data will be reported
        :param epoch_no: epoch number
        :param logging_interval: after how many batches data will be logged
        :return: mean loss for all batches
        """
        mean_loss = []
        model.train()
        for batch_idx, (batch, _) in enumerate(self.train_dataloader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            model_output = model(batch)
            loss = model.loss_fn(batch, model_output)
            loss.backward()
            optimizer.step()

            loss_float = loss.item()
            mean_loss.append(loss_float)
            experiment.log_metric("batch_train_loss", loss_float, step=epoch_no * batch_idx)

            if batch_idx % logging_interval == 0:
                print(f'=== train epoch {epoch_no}, [{batch_idx * len(batch)}/{len(self.train_dataloader.dataset)}] '
                      f'-> batch loss: {loss_float}')
        return sum(mean_loss) / len(mean_loss)

    def _val_step(self, model, experiment, epoch_no, logging_interval):
        """
        Function for performing validation step for model
        :param model:
        :param experiment:
        :param epoch_no:
        :param logging_interval:
        :return:
        """
        val_loss = []
        model.eval()
        for batch_idx, (batch, _) in enumerate(self.val_dataloader):
            model_output = model(batch)
            loss = model.loss_fn(batch, model_output)

            loss_float = loss.item()
            val_loss.append(loss_float)
            experiment.log_metric("batch_val_loss", loss_float, step=epoch_no * batch_idx)

            if batch_idx % logging_interval == 0:
                print(f'=== validation epoch {epoch_no}, '
                      f'[{batch_idx * len(batch)}/{len(self.train_dataloader.dataset)}]'
                      f'-> batch loss: {loss.item()} ===')

        return sum(val_loss) / len(val_loss)

    def find_best(self, model_type: HiveModelType, input_shape: Union[int, tuple],
                  learning_config: dict, n_trials=10) -> None:
        """
        Method for searching best architecture with oputa
        :param model_type:
        :param model_config:
        :param input_shape:
        :param learning_config:
        :param n_trials:
        """
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
        study.optimize(lambda op_trial: self._optuna_train_objective(
            op_trial, model_type, input_shape, learning_config), n_trials=n_trials)

        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f'    {key}:{value}')

    def _optuna_train_objective(self, trial: optuna.Trial, model_type: HiveModelType, input_shape: Union[int, tuple],
                                learning_config: dict) -> float:
        """
        Method for optuna objective
        :param trial: optuna.trial.Trial object
        :param model: model to be adjusted
        :param config: configuration for the model
        :return: final loss
        """
        model_config = build_optuna_ae_config(trial, input_shape)
        model = HiveModelFactory.build_model(model_type, input_shape, model_config)

        learning_config['learning_rate'] = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        learning_config['opitimizer'] = trial.suggest_categorical('optimizer',
                                                                  ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta'])

        experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            {**model.get_params(), **learning_config, **self.feature_config},
                                            ['optuna'])

        print(f'performing optuna train task on {self.device}:{torch.cuda.device_count()}'
              f' for model {type(model).__name__.lower()} with following config: {learning_config}')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        optimizer: Optimizer = _parse_optimizer(learning_config['optimizer'].get('type', 'Adam'))(
            model.parameters(), lr=learning_config['learning_rate'])
        batch_logging_interval = learning_config.get('logging_batch_interval', 10)
        train_epoch_loss = sys.maxsize
        for epoch in range(1, learning_config.get('epochs', 10) + 1):
            train_epoch_loss = self._train_step(model, optimizer, experiment, epoch, batch_logging_interval)
            experiment.log_metric('train_epoch_loss', train_epoch_loss, step=epoch)
            print(f'--- train epoch {epoch} end with train loss: {train_epoch_loss} ---')

            trial.report(train_epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return train_epoch_loss

    def train(self, model: BaseModel, config: dict) -> BaseModel:
        """
        Function for training model with specified config
        :param model: model to be trained
        :param config: configuration for learning process
        :rtype: BaseModel: trained model
        """
        best_val_loss = -1
        patience_counter = config.get('epoch_patience', 10)
        experiment = self._setup_experiment(f"{type(model).__name__.lower()}-{time.strftime('%Y%m%d-%H%M%S')}",
                                            {**model.get_params(), **config, **self.feature_config}, [])
        checkpoint_path = self.output_folder / f'{experiment.get_name()}-checkpoint.pth'

        print(f'performing train task on {self.device}:{torch.cuda.device_count()}'
              f' for model {checkpoint_path.stem.upper()} with following config: {config}')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)
        optimizer: Optimizer = _parse_optimizer(config['optimizer'].get('type', 'Adam'))(model.parameters(),
                                                                                         lr=config.get(
                                                                                             'learning_rate',
                                                                                             0.0001))
        batch_logging_interval = config.get('logging_batch_interval', 10)
        for epoch in range(1, config.get('epochs', 10) + 1):
            train_epoch_loss = self._train_step(model, optimizer, experiment, epoch, batch_logging_interval)
            experiment.log_metric('train_epoch_loss', train_epoch_loss, step=epoch)

            val_epoch_loss = self._val_step(model, experiment, epoch, batch_logging_interval)
            experiment.log_metric('val_epoch_loss', val_epoch_loss, step=epoch)

            print(f'--- train epoch {epoch} end with train loss: {train_epoch_loss} and val loss: {val_epoch_loss} ---')
            if val_epoch_loss < best_val_loss or best_val_loss == -1:
                print(f'*** model checkpoint at epoch {epoch} ***')
                best_val_loss = val_epoch_loss
                patience_counter = config.get('epoch_patience', 10)
                model_save(model, checkpoint_path, optimizer, epoch, train_epoch_loss)
            elif patience_counter == 0 or math.isnan(val_epoch_loss):
                print(f'___ early stopping at epoch {epoch} ___')
            else:
                patience_counter -= 1

        epoch, _ = model_load(checkpoint_path, model, optimizer)
        return model
