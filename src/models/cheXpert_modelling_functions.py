import pytorch_lightning as pl
import torch 
from torchmetrics.functional import accuracy
from torchmetrics import AUROC
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
#%% 

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
def get_params_to_update(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return params_to_update

## HELPER FUNCTION from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html  
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#%% Pytorh Lightning network
class BinaryClassificationTaskCheXpert(pl.LightningModule):
    def __init__(
            self, 
            model, 
            lr = 1e-3, 
            feature_extract = True, 
            reduce_lr_on_plateau = False,
            lr_scheduler_patience = None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.feature_extract = feature_extract
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.save_hyperparameters()
        self.train_auroc = AUROC(compute_on_step=False)
        self.val_auroc = AUROC(compute_on_step = False)
        self.test_auroc = AUROC(compute_on_step=False)

    #def forward(self, x):
    #    print(x)
    #    y_hat = torch.sigmoid(self.model(x.double()))
    #    return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x.double()))
        loss = F.binary_cross_entropy(y_hat, y.double())
        acc = accuracy(y_hat, y)
        metrics = {"train_loss": loss, 'train_acc': acc}
        self.log_dict(metrics)

        self.train_auroc(y_hat, y)
        self.log(
            'train_auroc', self.train_auroc, 
            on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x.double()))
        loss = F.binary_cross_entropy(y_hat, y.double())
        acc = accuracy(y_hat, y)
        
        metrics = {"val_loss": loss, 'val_acc': acc}
        self.log_dict(metrics)
        print("VALIDATING")
        self.val_auroc(y_hat, y)
        self.log(
            'val_auroc', self.val_auroc, 
            on_step = False, on_epoch = True)
        return metrics, y_hat

    def validation_epoch_end(self, val_step_outputs):
        preds = []
        for metrics, pred in val_step_outputs:
            print(f"metrics: {metrics}")
            print(f"pred: {pred}")
            preds.append(pred)
        print(f"preds:{preds}")    
        return metrics, preds

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x.double()))
        test_loss = F.binary_cross_entropy(y_hat, y.double())
        test_acc = accuracy(y_hat, y)
        
        metrics = {"test_loss": test_loss, 'test_acc': test_acc}
        self.log_dict(metrics)

        self.test_auroc(y_hat, y)
        self.log(
            'test_auroc', self.test_auroc, 
            on_step = False, on_epoch = True)
        return metrics

    def configure_optimizers(self):
        params_to_optimize = get_params_to_update(self.model, feature_extract = self.feature_extract)
        optimizer = torch.optim.SGD(params_to_optimize, lr = self.lr)
        optimizer_dict = {'optimizer': optimizer}
        if self.reduce_lr_on_plateau:
            lr_scheduler_config = {
                'scheduler': ReduceLROnPlateau(
                    optimizer, 
                    patience = self.lr_scheduler_patience),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'}
            optimizer_dict['lr_scheduler'] = lr_scheduler_config
        return optimizer_dict


