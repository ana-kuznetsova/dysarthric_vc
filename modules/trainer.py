from utils.utils import restore
import json
import torch
import sys
import os
import wandb
import numpy as np

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

        ###Restore from checkpoint if exists
        if os.listdir(self.config.runner.ckpt_path):
            restore(self.config, model, optimizer, scheduler)

        ########################################
        #######Speaker Encoder Training#########
        ########################################

        if self.config.model.model_name=='speaker_encoder':
            ep = 0
            step = 0
            prev_val_loss = 0
            val_count = 0
            criterion = criterion.to(device)

            if self.config.data.augment:
                num_speakers = self.config.model.num_speakers
                num_utter = self.config.model.num_utter*3
            else:
                num_speakers = self.config.model.num_speakers
                num_utter = self.config.model.num_utter
            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                epoch_train_loss = 0
                for batch in train_loader:
                    x = batch['x'].to(device)
                    spk_true = batch['spk_id'].to(device)
                    optimizer.zero_grad()
                    mini_steps = x.shape[0] // self.config.trainer.batch_size
                    outputs = torch.zeros(x.shape[0], self.config.model.feat_encoder_dim).to(device)
                    for mini_batch_idx in range(mini_steps):
                        start = mini_batch_idx*self.config.trainer.batch_size
                        end = min(start + self.config.trainer.batch_size, x.shape[0])
                        x_mini = x[start:end]
                        outputs[start:end,:] = model(x_mini, l2_norm=True)
                    out = outputs.view(num_speakers, num_utter, self.config.model.feat_encoder_dim)
                    loss = criterion(out, spk_true)
                    loss.backward()
                    optimizer.step()
                    step+=1
                    epoch_train_loss+=loss.data
                epoch_train_loss/=len(train_loader)

                print(f'> [Epoch]:{ep+1} [Train Loss]:{epoch_train_loss.data}')

                ##Validation loop
                val_num_utter = self.config.model.num_utter
                val_loss = 0
                val_change = True
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['x'].to(device)
                        spk_true = batch['spk_id'].to(device)
                        out = model(x, l2_norm=True)
                        out = out.view(num_speakers, val_num_utter, self.config.model.feat_encoder_dim)
                        loss = criterion(out, spk_true)
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
                else:
                    val_count+=1
                
                if val_count>=5:
                    if scheduler:
                        scheduler.step()
                    val_count=0
                    
                prev_val_loss = val_loss

                if ep%5==0:
                    sched_path = f"scheduler_{ep}.pth"
                    opt_path = f"optimizer_{ep}.pth"
                    model_path = f"model_{ep}.pth"

                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, model_path))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, opt_path))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, sched_path))

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


    def inference(self, train_loader, test_loader, model, device, parallel=False, out_dir='./'):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not parallel:
            print(f"> Using CUDA {device}")
            model = model.to(device)
        else:
            devices = list(self.config.runner.cuda_device)
            devices = [torch.device('cuda', i) for i in devices]
            print(f"> Using CUDA {devices}")
            model = model.to(device)
            model = torch.nn.DataParallel(model, device_ids=devices)
        
        restore(self.config, model, None, None, 'test')

        
        if self.config.model.model_name=='speaker_encoder':
            save_output = None
            save_spk = None

            print(f"> Generating training data...")
            for batch in train_loader:
                x = batch['x'].to(device)
                spk_true = batch['spk_id']
                mini_steps = x.shape[0] // self.config.trainer.batch_size
                outputs = torch.zeros(x.shape[0], self.config.model.feat_encoder_dim).to(device)
                for mini_batch_idx in range(mini_steps):
                    start = mini_batch_idx*self.config.trainer.batch_size
                    end = min(start + self.config.trainer.batch_size, x.shape[0])
                    x_mini = x[start:end]
                    outputs[start:end,:] = model(x_mini, l2_norm=True)
                
                outputs = outputs.detach().cpu().numpy()
                spk_true= spk_true.numpy()

                if not save_output:
                    save_output = outputs
                    save_spk = spk_true
                else:
                    save_output = np.concatenate([save_output, outputs], axis=0)
                    save_spk = np.concatenate([save_spk, spk_true], axis=0)

                
            np.save(os.path.join(out_dir, 'X_train.npy'), save_output)
            np.save(os.path.join(out_dir, 'Y_train.npy'), save_spk)

            save_output = None
            save_spk = None
            

            for batch in test_loader:
                x = batch['x'].to(device)
                spk_true = batch['spk_id']
                mini_steps = x.shape[0] // self.config.trainer.batch_size
                outputs = torch.zeros(x.shape[0], self.config.model.feat_encoder_dim).to(device)
                for mini_batch_idx in range(mini_steps):
                    start = mini_batch_idx*self.config.trainer.batch_size
                    end = min(start + self.config.trainer.batch_size, x.shape[0])
                    x_mini = x[start:end]
                    outputs[start:end,:] = model(x_mini, l2_norm=True)
                
                outputs = outputs.detach().cpu().numpy()
                spk_true = spk_true.numpy()

                if not save_output:
                    save_output = outputs
                    save_spk = spk_true
                else:
                    save_output = np.concatenate([save_output, outputs], axis=0)
                    save_spk = np.concatenate([save_spk, spk_true], axis=0)

            np.save(os.path.join(out_dir, 'X_test.npy'), save_output)
            np.save(os.path.join(out_dir, 'Y_test.npy'), save_spk)

            print(f'Saved inference results at {out_dir}')