import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn.functional as F
from deep_math.constants import PAD

OPTIMIZER = "Adam"
LR = 6e-4
BETA_COEFF_LOW = 0.9
BETA_COEFF_HIGH = 0.995
EPS = 1e-9
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(torchmetrics.Metric):
    """Accuracy Metric with a hack."""

    def __init__(self):
        super().__init__()
        self.add_state("n_char_total",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("n_char_correct",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")  # pylint: disable=not-callable

    def update(self, preds, target):
        # update metric states
        pred_max = preds.max(1)[1]
        target = target.contiguous().view(-1)
        non_pad_mask = target.ne(PAD)
        n_correct = pred_max.eq(target)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        non_pad_mask = target.ne(PAD)
        n_char = non_pad_mask.sum().item()
        self.n_char_total += n_char
        n_char = n_char if n_char > 1 else 1
        self.n_char_correct += n_correct

    def compute(self):
        # compute final result
        return self.n_char_correct / self.n_char_total


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.args = {}
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)
        self.beta_coeff_low = self.args.get("beta_coeff_low", BETA_COEFF_LOW)
        self.beta_coeff_high = self.args.get("beta_coeff_high", BETA_COEFF_HIGH)
        self.eps = self.args.get("eps", EPS)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps",
                                                   ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(),
                                         lr=self.lr,
                                         betas=(self.beta_coeff_low,
                                                self.beta_coeff_high),
                                         eps=self.eps)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        batch_qs, _, batch_as, _ = map(lambda x: x, batch)
        x, y = batch_qs, batch_as[:, 1:]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        batch_qs, _, batch_as, _ = map(lambda x: x, batch)
        x, y = batch_qs, batch_as[:, 1:]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc",
                 self.val_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        batch_qs, _, batch_as, _ = map(lambda x: x, batch)
        x, y = batch_qs, batch_as[:, 1:]
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def loss_fn(self, logits, y):
        y = y.contiguous().view(-1)
        loss = F.cross_entropy(logits, y, ignore_index=PAD, reduction="sum")
        return loss
