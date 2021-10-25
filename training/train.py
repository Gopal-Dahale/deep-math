import torch
import pytorch_lightning as pl
from deep_math.constants import MODELS, SIMPLE_LSTM, DATASETS, MINI
from deep_math.data.math_data_module import MathDataModule
from argparse import ArgumentParser
from deep_math.util import build_model, collate_fn
from deep_math import lit_models
import warnings

warnings.filterwarnings('ignore')


def _setup_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Name of the model",
        choices=MODELS,
        default=SIMPLE_LSTM,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to be used. Full or mini",
        choices=DATASETS,
        default=MINI,
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    model_type = args.model
    dataset_type = args.dataset

    collate_func = collate_fn(model_type)
    data = MathDataModule(collate_func, dataset_type)
    model = build_model(model_type)

    lit_model_class = lit_models.BaseLitModel
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint,
                                                         model=model)
    else:
        lit_model = lit_model_class(model=model)

    gpus = None  # CPU
    if torch.cuda.is_available():
        gpus = -1  # all available GPUs

    logger = pl.loggers.TensorBoardLogger("training/logs")
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss",
                                                         mode="min",
                                                         patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
        monitor="val_loss",
        mode="min")

    callbacks = [early_stopping_callback, model_checkpoint_callback]

    trainer = pl.Trainer(gpus=gpus,
                         fast_dev_run=True,
                         max_epochs=1,
                         logger=logger,
                         callbacks=callbacks,
                         weights_save_path='training/logs',
                         weights_summary='full')

    trainer.tune(
        model=lit_model,
        datamodule=data,
    )
    trainer.fit(model=lit_model, datamodule=data)
    trainer.test(model=lit_model, datamodule=data)


if __name__ == "__main__":
    main()