from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

def main():
    args = OmegaConf.load('rnnt_args.yaml')
    trainer = BrainToTextDecoder_Trainer(args)
    metrics = trainer.train()
if __name__ == '__main__':
    main()