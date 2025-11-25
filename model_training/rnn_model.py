import torch 
from torch import nn

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits
        
class RNNT(nn.Module):
    def __init__(self, input_dim, enc_dim, pred_dim, joint_dim, num_classes, n_days = 1, input_dropout = 0.0, patch_size = 0, patch_stride = 0):
        '''
        Recurrent Neural Network Transducer (RNN-T).
        - Encoder consumes acoustic/neural features
        - Predictor consumes previous output symbols (via embedding)
        - Joint combines both streams
        '''
        super(RNNT, self).__init__()
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.joint_dim = joint_dim
        self.num_classes = num_classes  # includes blank id (typically 0)
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific projection (parity with GRUDecoder)
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.input_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.input_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(self.input_dropout)

        # If using strided input concatenation, expand encoder input size accordingly
        self.encoder_input_size = self.input_dim
        if self.patch_size > 0:
            self.encoder_input_size *= self.patch_size

        # Encoder and Predictor
        self.encoder = nn.GRU(self.encoder_input_size, self.enc_dim, batch_first = True)
        self.predictor = nn.GRU(self.pred_dim, self.pred_dim, batch_first = True)

        # Label embedding for predictor input (use class ids incl. blank/SOS)
        self.label_embedding = nn.Embedding(self.num_classes, self.pred_dim)

        # Projections and Joint network
        self.enc_proj = nn.Linear(self.enc_dim, self.joint_dim)
        self.pred_proj = nn.Linear(self.pred_dim, self.joint_dim)
        self.joint_activation = nn.Tanh()
        self.joint_out = nn.Linear(self.joint_dim, self.num_classes)

        # Learnable initial states
        self.h0_enc = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.enc_dim)))
        self.h0_pred = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.pred_dim)))

    def _apply_day_projection(self, x, day_idx):
        # x: [B, T, D]
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        day_weights = day_weights.to(x.dtype)
        day_biases = day_biases.to(x.dtype)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        return x

    def _apply_patching(self, x):
        # Optionally perform input concatenation using sliding windows
        if self.patch_size <= 0:
            return x
        x = x.unsqueeze(1)                      # [B, 1, T, D]
        x = x.permute(0, 3, 1, 2)               # [B, D, 1, T]
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [B, D, 1, P, K]
        x_unfold = x_unfold.squeeze(2)          # [B, D, P, K]
        x_unfold = x_unfold.permute(0, 2, 3, 1) # [B, P, K, D]
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)  # [B, P, K*D]
        return x

    def encode(self, x, day_idx):
        # x: [B, T, D]
        x = self._apply_day_projection(x, day_idx)
        x = self._apply_patching(x)
        h0 = self.h0_enc.expand(1, x.shape[0], self.enc_dim).contiguous()
        enc_out, _ = self.encoder(x, h0)
        return enc_out  # [B, T', E]

    def _joint(self, enc, pred):
        # enc: [B, T, E]; pred: [B, U, P]
        e = self.enc_proj(enc).unsqueeze(2)     # [B, T, 1, J]
        p = self.pred_proj(pred).unsqueeze(1)   # [B, 1, U, J]
        z = self.joint_activation(e + p)        # [B, T, U, J]
        logits = self.joint_out(z)              # [B, T, U, C]
        return logits

    def forward(self, x, day_idx, targets_with_sos):
        '''
        x: [B, T, D]; day_idx: [B]; targets_with_sos: [B, U+1] where first token is SOS (typically blank=0)
        Returns: logits [B, T', U+1, C]
        '''
        enc = self.encode(x, day_idx)
        # Embed predictor inputs
        y_emb = self.label_embedding(targets_with_sos)  # [B, U+1, P]
        h0 = self.h0_pred.expand(1, x.shape[0], self.pred_dim).contiguous()
        pred_out, _ = self.predictor(y_emb, h0)         # [B, U+1, P]
        logits = self._joint(enc, pred_out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, x, day_idx, blank_id = 0, max_symbols_per_step = 30):
        '''
        Simple RNNT greedy decoding for a single batch (loops over batch elements).
        Returns list of predicted int sequences (without blanks).
        '''
        x = x.float()
        enc = self.encode(x, day_idx)  # [B, T, E]
        B, T, _ = enc.shape
        results = []
        for b in range(B):
            enc_b = enc[b:b+1]  # [1, T, E]
            # predictor state and last symbol (SOS=blank)
            y = torch.tensor([[blank_id]], device=enc_b.device, dtype=torch.long)  # [1, 1]
            y_emb = self.label_embedding(y)  # [1,1,P]
            h = self.h0_pred.expand(1, 1, self.pred_dim).contiguous()
            pred_out, h = self.predictor(y_emb, h)  # [1,1,P]
            tokens = []
            for t in range(T):
                # compute joint for current time step with current predictor output
                e = self.enc_proj(enc_b[:, t:t+1, :]).unsqueeze(2)  # [1,1,1,J]
                p = self.pred_proj(pred_out)                       # [1,1,P] -> [1,1,J] after proj
                p = p.unsqueeze(1)                                  # [1,1,1,J]
                z = self.joint_activation(e + p)
                logit = self.joint_out(z).squeeze(1).squeeze(1)    # [1,C]
                c = torch.argmax(logit, dim=-1).item()
                sym_per_step = 0
                while c != blank_id and sym_per_step < max_symbols_per_step:
                    tokens.append(c)
                    # feed back into predictor
                    y = torch.tensor([[c]], device=enc_b.device, dtype=torch.long)
                    y_emb = self.label_embedding(y)
                    pred_out, h = self.predictor(y_emb, h)
                    # recompute joint with updated predictor state
                    p = self.pred_proj(pred_out).unsqueeze(1)
                    z = self.joint_activation(e + p)
                    logit = self.joint_out(z).squeeze(1).squeeze(1)
                    c = torch.argmax(logit, dim=-1).item()
                    sym_per_step += 1
            results.append(tokens)
        return results





class RNNT_2(nn.Module):
    def __init__(self, input_dim, enc_dim, pred_dim, joint_dim, num_classes):
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.joint_dim = joint_dim
        self.num_classes = num_classes

        self.encoder = nn.GRU(self.input_dim, self.enc_dim)
        self.predictor = nn.GRU(self.num_classes, self.pred_dim)

        self.enc_proj = nn.Linear()
        self.pred_proj = nn.Linear()

        self.joint = nn.Sequential(
            nn.Linear(self.joint_dim, self.joint_dim)
        )