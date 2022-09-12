from distutils.command.config import config
import json
import torch
import sys

class Trainer():
    def __init__(self, configs):
        super(Trainer, self).__init__()

        self.epoch = configs.trainer.epoch
        self.batch_size = configs.trainer.batch_size
        self.data_parallel = configs.trainer.data_parallel
        self.data_config = configs.data
        self.config = configs
    
    def train(self, train_loader, 
              val_loader, model, criterion,
              optimizer,
              scheduler,
              device, 
              parallel=False):
        
        if not parallel:
            device = "cuda"
            print(f"> Using CUDA {device}")

        model = model.to(device)

        if self.config.model.model_name=='speaker_encoder':
            step = 0
            prev_val_loss = 0
            while ep < config.trainer.epoch:
                print(f"Starting [epoch]:{ep}/{config.trainer.epoch}")
                for batch in train_loader:
                    x = batch['x'].to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    out = out.view(config.model.num_speakers, config.model.num_utter, config.model.feat_encoder_dim)
                    loss = criterion(out)
                    loss.backward()
                    scheduler.step()
                    step+=1
                    if step%2000==0:
                        print(f"> Train loss after {step} steps")
                        print(f'>[Loss]:{loss.data:.5f}')
                        if config.runner.wandb:
                            wandb.log({"train_loss": loss})

                ##Validation loop
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['x'].to(device)
                        out = model(x)
                        out = out.view(config.model.num_speakers, config.model.num_utter, config.model.feat_encoder_dim)
                        loss = criterion(out)
                        val_loss+=loss.data
                val_loss = val_loss/len(val_loader)
                print(f"> Val loss after {ep} epochs")
                print(f'> [Loss]:{val_loss:.5f}')
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(config.runner.ckpt_path, "best_model.pth"))
                    torch.save(scheduler.state_dict(), os.path.join(config.runner.ckpt_path, "scheduler.pth"))
                prev_val_loss = val_loss
                ep+=1
                


    def inference(self):
        pass