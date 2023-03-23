
from argparse import ArgumentParser
import os
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import data_processing
# Models
from models.model import DeepSpeech2

class SpeechModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=2)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), 
                                           lr = self.config.learning_rate,
                                           betas=(0.9, 0.999),
                                           weight_decay = 1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.50, patience=6)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': 'val_loss'}

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        output = self(spectrograms)  # (batch, time, n_class)
        output = output.transpose(0, 1) # (time, batch, n_class)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log("train_loss", loss)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {'val_loss': loss}
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

def checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename = 'sr-betas(0.9,0.999)-wd(1e-6)--{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    
def main(args):
    train_dataset = torchaudio.datasets.LIBRISPEECH(args.ds_loc, url="train-clean-100", download=True)
    val_dataset = torchaudio.datasets.LIBRISPEECH(args.ds_loc, url="test-clean", download=True)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'))
    val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'val'))
    # Prepare Model
    model = DeepSpeech2(
        n_cnn_layers = args.n_cnn_layers, 
        n_rnn_layers = args.n_rnn_layers,
        rnn_dim = args.rnn_dim,
        n_class = args.n_class,
        n_feats = args.n_feats,
        stride = args.stride,
        dropout = args.dropout)
    print('Number of Model Parameters', sum([param.nelement() for param in model.parameters()]))
    print(f"Using {args.device}")

    speech_module = SpeechModule(model, args)

    logger = TensorBoardLogger(args.logdir, name='speech recognition')

    trainer = Trainer(
        accelerator = args.device, 
        devices = args.devices, 
        logger=logger, 
        max_epochs = args.epochs, 
        gradient_clip_val=1.0,
        callbacks= [checkpoint_callback(args)],
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.fit(speech_module, train_loader, val_loader)
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    # Model hyper parameters
    parser.add_argument('--n_cnn_layers', default=3, type=int, help='Number of CNN layers')
    parser.add_argument('--n_rnn_layers', default=5, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_dim', default=256, type=int, help='RNN dimension')
    parser.add_argument('--n_class', default=29, type=int, help='Number of character classes')
    parser.add_argument('--n_feats', default=128, type=int, help='Number of features')
    parser.add_argument('--stride', default=2, type=int, help='Number of stride in CNN layer')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate of Model')
    parser.add_argument('--learning_rate', default= 1e-5, type=float, help='Learning Rate')

    # general
    parser.add_argument('--epochs', default=40, type=int, help='number of total epochs to run')
    parser.add_argument('--train_batch_size', default=8, type=int, help='number of total epochs to run')
    parser.add_argument('--val_batch_size', default=4, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default="gpu", type=str, help='device for acceleration')
    parser.add_argument('--devices', default=1, type=int, help='number of gpu devices')
    
    # dir and path for dataset, models, and logs
    parser.add_argument('--ds_loc', default='./LibriSpeech', type=str,
                        help='where to save dataset')    
    parser.add_argument('--resume_from_checkpoint', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--checkpoint_path', default='./checkpoint/', type=str,
                        help='path to save checkpoint')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str,
                        help='path to save logs')
    
    args = parser.parse_args()

    if args.checkpoint_path:
       if not os.path.isdir(os.path.dirname(args.checkpoint_path)):
           raise Exception("the directory for path {} does not exist".format(args.checkpoint_path))

    main(args)