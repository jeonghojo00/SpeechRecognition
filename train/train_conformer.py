from argparse import ArgumentParser
import configparser
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.conformer import Conformer
from scorer import wer, cer
from decode import GreedyCTCDecoder
from preprocess import collate_fn


class Conformer_SpeechModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        # n_class is len(LABELS)+1 due to the count of blank_id as an extra other than LABELS
        self.criterion = nn.CTCLoss(blank=int(config['MODEL']['n_class'])-1, zero_infinity=True)
        self.learning_rate = float(config['TRAIN']['learning_rate'])
        self.beta_1 = float(config['TRAIN']['beta_1'])
        self.beta_2 = float(config['TRAIN']['beta_2'])
        self.weight_decay = float(config['TRAIN']['weight_decay'])
        self.greedy_ctc_decoder = GreedyCTCDecoder(
            labels = list(config['DEFAULT']['LABELS']), blank = int(config['MODEL']['n_class'])-1)

    def forward(self, x, x_lens):
        output_probs, output_lens = self.model(x, x_lens)
        return output_probs, output_lens

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), 
                                           lr     = self.learning_rate,
                                           betas  =(self.beta_1, self.beta_2),
                                           weight_decay = self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.50, patience=6)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        spectrograms, labels, spec_lens, label_lens = batch
        output_probs, output_lens = self(spectrograms, spec_lens) # outputs: (batch, time, n_class)
        loss = self.criterion(output_probs.transpose(0, 1), labels, output_lens, label_lens)       
        
        self.log('train_loss', loss)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_mean', train_loss_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        spectrograms, labels, input_lens, label_lens = batch
        outputs, output_lens = self(spectrograms, input_lens) # outputs: (batch, time, n_class)
        loss = self.criterion(outputs.transpose(0, 1), labels, output_lens, label_lens)    
        
        cer, wer = self.get_cer_wer(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_wer', wer)
        self.log('val_cer', cer)
        return {'val_loss': loss, 'val_wer': wer, 'val_cer': cer}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_wer_mean = torch.stack([x['val_wer'] for x in outputs]).mean()
        val_cer_mean = torch.stack([x['val_cer'] for x in outputs]).mean()
        self.log('val_loss_mean', val_loss_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_wer_mean', val_wer_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cer_mean', val_cer_mean, on_epoch=True, prog_bar=True, logger=True)
        self.scheduler.step(val_loss_mean)
    
    def get_cer_wer(self, outputs, labels):
        cer_list = list()
        wer_list = list()
        for i in range(len(outputs)):
            decoded_pred = ' '.join(self.greedy_ctc_decoder(emission = outputs[i], predicted = True))
            decoded_target = ' '.join(self.greedy_ctc_decoder(emission = labels[i], predicted = False))
            cer_list.append(cer(decoded_target, decoded_pred))
            wer_list.append(wer(decoded_target, decoded_pred))

        avg_cer = sum(cer_list) / len(cer_list)
        avg_wer = sum(wer_list) / len(wer_list)
        return torch.tensor(avg_cer), torch.tensor(avg_wer)

def checkpoint_callback(checkpoint_path):
    return ModelCheckpoint(
        dirpath=checkpoint_path,
        filename = 'conformer-lr(1e-5)-dropout(0.1)-betas(0.9,0.98)-wd(1e-9)--{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    
def main(config):
    train_dataset = torchaudio.datasets.LIBRISPEECH(config['DEFAULT']['data_location'], url="train-clean-100", download=True)
    val_dataset   = torchaudio.datasets.LIBRISPEECH(config['DEFAULT']['data_location'], url="test-clean", download=True)

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = DataLoader(dataset = train_dataset,
                            batch_size= int(config['TRAIN']['train_batch_size']),
                            shuffle   = True,
                            collate_fn= lambda x: collate_fn(x, train=True),
                            **kwargs)
    val_loader = DataLoader(dataset   = val_dataset,
                            batch_size= int(config['TRAIN']['val_batch_size']),
                            shuffle   = False,
                            collate_fn= lambda x: collate_fn(x, train=False),
                            **kwargs)

    # Prepare Model
    model_params = config['MODEL']
    model = Conformer(
        num_classes         = int(model_params['n_class']),
        n_feats             = int(model_params['n_feats']),
        conv_sampling       = model_params['conv_sampling'],
        n_encoders          = int(model_params['n_encoders']),
        embed_dim           = int(model_params['embed_dim']),
        ff_expansion_factor = int(model_params['ff_expansion_factor']),
        mha_heads           = int(model_params['mha_heads']),
        conv_kernel_size    = int(model_params['conv_kernel_size']),
        dropout_p           = float(model_params['dropout_p']),
        lstm_layers         = int(model_params['lstm_layers'])
        )
 
    print('Number of Model Parameters', sum([param.nelement() for param in model.parameters()]))
    print(f"Using {config['DEFAULT']['device']}")

    speech_module = Conformer_SpeechModule(model, config)

    logger = TensorBoardLogger(config['DEFAULT']['logdir'], name='conformer')

    trainer = Trainer(
        accelerator = config['DEFAULT']['device'], 
        devices = int(config['DEFAULT']['devices']), 
        logger = logger, 
        max_epochs = int(config['TRAIN']['epochs']), 
        gradient_clip_val = 1.0,
        callbacks = [checkpoint_callback(str(config['DEFAULT']['checkpoint_path']))],
        resume_from_checkpoint = config['DEFAULT']['resume_from_checkpoint']
    )

    trainer.fit(speech_module, train_loader, val_loader)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    # Data Preparation
    parser.add_argument('--config_file', default='conformer_config.ini', type=str, help='Configuration ini file path')
    args = parser.parse_args()
    
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(args.config_file)
    
    checkpoint_path = config['DEFAULT']['checkpoint_path']
    if checkpoint_path:
       if not os.path.isdir(os.path.dirname(checkpoint_path)):
           raise Exception("the directory for path {} does not exist".format(checkpoint_path))

    main(config)