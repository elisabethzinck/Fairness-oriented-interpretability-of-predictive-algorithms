

def get_n_total_parameters(pytorch_model):
    n_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    return n_params

def get_n_hidden_list(params):
    """Extract list of hidden units from parameter dict"""
    n_hidden_list = []
    for i in range(params['n_layers']):
        name = 'n_hidden_' + str(i)
        n_hidden_list.append(params[name])
    return n_hidden_list


import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

# Soeren suggestion
# From https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference
class ClassificationTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
