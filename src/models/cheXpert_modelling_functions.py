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
            model = None,
            lr = 1e-3, 
            feature_extract = True, 
            reduce_lr_on_plateau = False,
            lr_scheduler_patience = None,
            optimizer = 'Adam', 
            num_classes = 1, 
            weight_decay = 0,
            dropout = 0):

        super().__init__()
        self.lr = lr
        self.feature_extract = feature_extract
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.dropout = dropout

        if model is None:
            self.model = self.initialize_model()
        else:
            self.model = model
        self.save_hyperparameters()
        self.train_auroc = AUROC(
            compute_on_step=False, 
            num_classes = self.num_classes, 
            average = 'micro')
        self.val_auroc = AUROC(
            compute_on_step = False, 
            num_classes = self.num_classes,
            average = 'micro')

    def initialize_model(self):
        model = torch.hub.load(
                'pytorch/vision:v0.10.0', 
                'densenet121', 
                pretrained = True, 
                drop_rate = self.dropout)

        set_parameter_requires_grad(model, 
            feature_extracting=self.feature_extract)

        in_features_classifier = model.classifier.in_features
        model.classifier = torch.nn.Linear(
            in_features_classifier, self.num_classes)

        model = model.double() # To ensure compatibilty with dataset
        return model

    def forward(self, batch): 
        x, y = batch
        forwarded_x = self.model(x.double())
        return forwarded_x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x.double()))
        loss = F.binary_cross_entropy(y_hat, y.double())
        acc = accuracy(y_hat, y)
        metrics = {"train_loss": loss, 'train_acc': acc}
        self.log_dict(metrics, on_step = False, on_epoch = True)

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
        self.log_dict(metrics, on_step = False, on_epoch = True)

        self.val_auroc(y_hat, y)
        self.log(
            'val_auroc', self.val_auroc, 
            on_step = False, on_epoch = True)
        return metrics

    def configure_optimizers(self):
        params_to_optimize = get_params_to_update(self.model, feature_extract = self.feature_extract)
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr = self.lr,
                weight_decay = self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr = self.lr,
                weight_decay = self.weight_decay
                )
        else:
            raise ValueError(f'optimizer `{self.optimizer}` not implemented.')
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


