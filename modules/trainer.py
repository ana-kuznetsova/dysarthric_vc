from distutils.command.config import config
import json
import torch
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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
            print(f"> Using CUDA {device}")
            model = model.to(device)
        else:
            devices = list(self.config.runner.cuda_device)
            devices = [torch.device('cuda', i) for i in devices]
            print(f"> Using CUDA {devices}")
            model = model.to(device)
            model = torch.nn.DataParallel(model, device_ids=devices)

        if self.config.model.model_name=='speaker_encoder':
            ep = 0
            step = 0
            prev_val_loss = 0
            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                for batch in train_loader:
                    x = batch['x'].to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    out = out.view(self.config.model.num_speakers, self.config.model.num_utter, self.config.model.feat_encoder_dim)
                    loss = criterion(out)
                    loss.backward()
                    optimizer.step()
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
                        out = out.view(self.config.model.num_speakers, self.config.model.num_utter, self.config.model.feat_encoder_dim)
                        loss = criterion(out)
                        val_loss+=loss.data
                val_loss = val_loss/len(val_loader)
                print(f"> Val loss after {ep} epochs")
                print(f'> [Loss]:{val_loss:.5f}')
                if config.runner.wandb:
                    wandb.log({"val_loss": loss})
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_model.pth"))
                    torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, "scheduler.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, "optimizer.pth"))
                prev_val_loss = val_loss
                ep+=1
                


    def inference(self):
        pass