from distutils.command.config import config
import json
import torch
import sys
import os
import wandb

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
        

        ########################################
        #######Speaker Encoder Training#########
        ########################################

        if self.config.model.model_name=='speaker_encoder':
            ep = 0
            step = 0
            prev_val_loss = 0
            criterion = criterion.to(device)
            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                epoch_train_loss = 0
                for batch in train_loader:
                    x = batch['x'].to(device)
                    spk_true = batch['spk_id'].to(device)
                    optimizer.zero_grad()
                    out = model(x, l2_norm=True)
                    out = out.view(self.config.model.num_speakers, self.config.model.num_utter, self.config.model.feat_encoder_dim)
                    loss = criterion(out, spk_true)
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    step+=1
                    epoch_train_loss+=loss.data
                epoch_train_loss/=len(train_loader)

                print(f'> [Epoch]:{ep+1} [Train Loss]:{epoch_train_loss.data}')

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
                print(f'> [Epoch]:{ep+1} [Valid Loss]:{val_loss.data}')
                if self.config.runner.wandb:
                    wandb.log({"train_loss": epoch_train_loss.data, "val_loss": val_loss})
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_model.pth"))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, "scheduler.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, "optimizer.pth"))
                prev_val_loss = val_loss
                ep+=1


        ########################################
        #### General Encoder Training (no MI)###
        ########################################

        if self.config.model.model_name=='general_encoder':
            ep = 0
            step = 0
            prev_val_loss = 0
            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                epoch_train_loss_total = 0
                epoch_train_loss_1 = 0
                epoch_train_loss_2 = 0
                for batch in train_loader:
                    x = batch['x'].to(device)
                    p = batch['p'].to(device)
                    spk_true = batch['spk_id'].to(device)
                    optimizer.zero_grad()
                    outs = model(x, p)
                    loss, l1, l2 = criterion(outs['feats'], outs['proj'], outs['spk_cls_out'], spk_true)
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    step+=1
                    epoch_train_loss+=loss.data
                    epoch_train_loss_1+=l1.data
                    epoch_train_loss_2+=l2.data

                epoch_train_loss/=len(train_loader)
                epoch_train_loss_1/=len(train_loader)
                epoch_train_loss_2/=len(train_loader)

                print(f'> [Epoch]:{ep+1} [Train Loss]:{epoch_train_loss.data}')
                if self.config.runner.wandb:
                    wandb.log({"train_loss": epoch_train_loss.data,
                                "loss_1": epoch_train_loss_1, "loss2":epoch_train_loss_2})

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
                print(f'> [Epoch]:{ep+1} [Valid Loss]:{val_loss.data}')
                if self.config.runner.wandb:
                    wandb.log({"val_loss": val_loss})
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_model.pth"))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, "scheduler.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, "optimizer.pth"))
                prev_val_loss = val_loss
                ep+=1


    def inference(self):
        pass