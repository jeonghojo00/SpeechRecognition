import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from models.speech_transformer import SpeechTransformer
from models.modules.loss import Joint_CTC_CrossEntropy_Loss
from utils import add_zeros

def main(args):
    device = args.device
    
    model = SpeechTransformer(
        src_n_feats = args.n_feats,
        src_seq_len = args.src_seq_len, 
        enc_layers = args.enc_layers, 
        embed_dim = args.embed_dim, 
        expansion_factor = args.expansion_factor, 
        n_heads = args.n_heads,
        n_class=args.n_class, 
        trg_seq_len=args.trg_seq_len, 
        dec_layers=args.dec_layers, 
        dropout_rate=args.dropout_rate, 
        device=args.device)
    
    criterion = Joint_CTC_CrossEntropy_Loss(
        num_classes = args.n_class,                       # the number of classfication
        dim = args.dim,                                   # dimension of caculation loss
        reduction = args.reduction,                       # reduction method ['sum', 'mean', 'elementwise_mean']
        smoothing = args.smoothing,                       # ratio of smoothing (confidence = 1.0 - smoothing)
        ctc_weight = args.ctc_weight,                     # ratio of ctc loss (cross entropy loss ratio = 1.0 - ctc loss ratio)
        blank_id = args.blank_id,
        architecture = args.architecture)
    
    batch_size = 2
    n_feats = args.n_feats
    mel_seq_lens = 1300
    label_seq_lens = 300

    spec = torch.randn(batch_size, 1, n_feats, mel_seq_lens).to(device)
    labels = torch.randint(low=0, high=29, size=(batch_size, label_seq_lens), dtype=torch.long).to(device)
    input_lens = torch.tensor([1200, 1300], dtype=torch.long).to(device)
    label_lens = torch.tensor([200, 300], dtype=torch.long).to(device)
    
    print("\nTraining....................")
    model.train()
    output, enc_log_probs, enc_output_lens = model(spec, input_lens, labels[:, :-1]) # Insert labels except the last ones. Labels include starting token with 0 index
    print(f"Output: {output.shape}")
    print(f"ENC output: {enc_log_probs.shape}")
    print(f"ENC output lens: {enc_output_lens}")
    
    # Check for losses for Training
    output_softmax = F.log_softmax(output, dim=2) # (batch, time, n_class) after softmax
    print(f"output Softmax  : {output_softmax.shape}")
        
    loss, ctc_loss, ce_loss = criterion(encoder_log_probs = enc_log_probs,
                                    encoder_output_lens = enc_output_lens,
                                    output_softmax = output_softmax,
                                    target = labels[:, 1:],       # Now the first starting tokens are removed
                                    target_lens = label_lens-1)   # labels lens are reduced by 1 by removing starting tokens
    print(f"loss: {loss.item():.3}, ctc_loss: {ctc_loss.item():.3}, ce_loss: {ce_loss.item():.3}")
    
    print("\nValidation....................")
    model.eval()
    torch.no_grad()
    output_valid, enc_log_probs_valid, enc_output_lens_valid = model.recognize(spec, input_lens)
    print(f"Output: {output_valid.shape}")
    print(f"ENC output: {enc_log_probs.shape}")
    print(f"ENC output lens: {enc_output_lens}")

    new_labels = add_zeros(labels, args.trg_seq_len, device)
    # Check for losses for Validation
    output_valid_softmax = F.log_softmax(output_valid, dim=2) # (batch, time, n_class) after softmax

    loss, ctc_loss, ce_loss = criterion(encoder_log_probs = enc_log_probs_valid,
                                        encoder_output_lens = enc_output_lens_valid,
                                        output_softmax = output_valid_softmax,
                                        target = new_labels[:, 1:],
                                        target_lens = label_lens-1)

    print(f"loss: {loss.item():.3}, ctc_loss: {ctc_loss.item():.3}, ce_loss: {ce_loss.item():.3}")
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    # Model hyper parameters
    ## For Input Encoder
    parser.add_argument('--n_feats'          , default=80, type=int, help='number of features for input')
    parser.add_argument('--conv_out_channels', default=64, type=int, help='number of channels to extract features in input encoder')
    parser.add_argument('--conv_kernel_size' , default=3 , type=int, help='kernel size for CNN in input encoder')
    parser.add_argument('--conv_stride'      , default=2 , type=int, help='stride size for CNN in input encoder')
    
    ## Fpr Transformer
    parser.add_argument('--embed_dim'        , default=256 , type=int  , help='embedding dimension for transformer')
    parser.add_argument('--src_seq_len'      , default=700 , type=int  , help='maximum sequence length for encoder')
    parser.add_argument('--trg_seq_len'      , default=500 , type=float, help='maximum sequence length for decoder')
    parser.add_argument('--enc_layers'       , default= 4  , type=int  , help='number of encoder layers')
    parser.add_argument('--dec_layers'       , default= 4  , type=int  , help='number of decoder layers')
    parser.add_argument('--expansion_factor' , default= 8  , type=int  , help='expansion ratio for encoder/decoder block')
    parser.add_argument('--n_heads'          , default= 4  , type=int  , help='number of heads in attention')
    parser.add_argument('--dropout_rate'     , default= 0.1, type=float, help='dropout rate of transforemr')
    parser.add_argument('--n_class'          , default= 29 , type=int  , help='Number of classes to label for output')

    ## Fpr Transformer
    parser.add_argument('--ignore_index', default= 0            , type=int  , help='index to ignore when calcuating loss')
    parser.add_argument('--dim'         , default= -1           , type=int  , help='dimension of caculation loss')
    parser.add_argument('--reduction'   , default='mean'        , type=str  , help="reduction method ['sum', 'mean', 'elementwise_mean']")
    parser.add_argument('--smoothing'   , default= 0.0          , type=float, help='ratio of smoothing (confidence = 1.0 - smoothing)')
    parser.add_argument('--ctc_weight'  , default= 0.7          , type=float, help='ratio of ctc loss (cross entropy loss ratio = 1.0 - ctc loss ratio)')
    parser.add_argument('--blank_id'    , default= 0            , type=int  , help='blank index to ignore during calculating ctc loss')
    parser.add_argument('--architecture', default= 'transformer', type=str  , help="model name ['tranformer', 'ds2']")
    
    # Common
    parser.add_argument('--device', default="cpu", type=str, help='device for acceleration')
    parser.add_argument('--devices', default=1, type=int, help='number of gpu devices')

    args = parser.parse_args()

    main(args)
