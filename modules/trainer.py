from utils.utils import restore
import json
import torch
import sys
import os
import wandb
import numpy as np
from tqdm import tqdm
from utils.utils import optimizer_to, shuffle_tensor
from modules.losses import CLUB
from modules.decoder import Interface
from utils.feat_extractor import FeatureExtractor

class Trainer():
    def __init__(self, configs):
        super(Trainer, self).__init__()

        self.epoch = configs.trainer.epoch
        self.batch_size = configs.trainer.batch_size
        self.data_parallel = configs.runner.data_parallel
        self.data_config = configs.data
        self.config = configs
    
    
    def train(self, train_loader, 
              val_loader, model, criterion,
              optimizer,
              scheduler,
              device, 
              parallel=False,
              **kwargs):


        if not parallel:
            print(f"> Using CUDA {device}")
            model = model.to(device)
        else:
            devices = list(self.config.runner.cuda_device)
            devices = [torch.device('cuda', i) for i in devices]
            print(f"> Using CUDA {devices}")
            model = model.to(device)
            model = torch.nn.DataParallel(model, device_ids=devices)
        criterion = criterion.to(device)

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
        #### Joint VC Training###
        ########################################

        if (self.config.model.model_name=='joint_vc') or (self.config.model.model_name=='dysarthric_vc'):
            model_name = self.config.model.model_name
            if self.config.model.restore_epoch:
                ep = self.config.runner.restore_epoch
                optimizer_to(optimizer, device)
            else:
                ep = 0
            step = 0
            prev_val_loss = 0
            no_improvement = 0
            interface = Interface()
            interface = interface.to(device)
            if self.config.model.interface:
                interface.load_state_dict(torch.load(self.config.model.interface))
            
            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                epoch_train_loss = 0
                epoch_train_loss_1 = 0
                epoch_train_loss_2 = 0
                epoch_train_mse = 0
           
                model.train()
                for i, batch in enumerate(train_loader):
                    x = batch['x'].to(device)
                    text = batch['text'].to(device)
                    target = batch["target"].to(device)
                    spk_true = batch['spk_id'].to(device)
                    d_labels = batch['d_labels'].to(device)
                    optimizer.zero_grad()
                    outs = model(x, text, target, interface)
    
                    if model_name=='joint_vc':
                        loss, l1, l2, mse =  criterion(x,  spk_true, outs, ep)
                    else:
                        loss, l1, l2, mse = criterion(x, d_labels, outs)

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
                epoch_train_mse/=len(train_loader)

                ##Validation loop
                val_loss = 0
                val_loss_1 = 0
                val_loss_2 = 0
                val_loss_mse = 0
        
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['x'].to(device)
                        text = batch['text'].to(device)
                        target = batch["target"].to(device)
                        spk_true = batch['spk_id'].to(device)
                        d_labels = batch['d_labels'].to(device)
                        optimizer.zero_grad()
                        outs = model(x, text, target, interface)
    
                        if model_name=='joint_vc':
                            loss, l1, l2, mse =  criterion(x,  spk_true, outs, ep)
                        else:
                            loss, l1, l2, mse = criterion(x, d_labels, outs)
                        val_loss+=loss.data
                        val_loss_1+=l1.data
                        val_loss_2+=l2.data
                        val_loss_mse+=mse.data

                val_loss/=len(val_loader)
                val_loss_1/=len(val_loader)
                val_loss_2/=len(val_loader)
                val_loss_mse/=len(val_loader)

                print(f'> Missed validation batches in epoch {ep}: {missed_val_batches}, missed files {missed_val_fnames}')
                print(f'> [Epoch]:{ep+1} [Train Loss]:{epoch_train_loss.data}')
                print(f'> [Epoch]:{ep+1} [Valid Loss]:{val_loss.data}')
                print(f'> Missed batches in epoch {ep}: {missed_batches}, missed files {missed_fnames}')
                print(f'> Missed validation batches in epoch {ep}: {missed_val_batches}, missed files {missed_val_fnames}')

                if self.config.runner.wandb:
                    wandb.log({"train_loss": epoch_train_loss.data,
                               "val_loss": val_loss.data, "l1_rc":epoch_train_loss_1,
                                "ce_loss":epoch_train_loss_2, "mse_loss": epoch_train_mse,
                                "val_l1_rc_loss":val_loss_1, "val_mse_loss":val_loss_mse,
                                "val_ce_loss":val_loss_2})
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_model.pth"))
                    torch.save(interface.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_interface.pth"))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, "optimizer.pth"))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, "scheduler.pth"))
                else:
                    no_improvement+=1

                prev_val_loss = val_loss

                if scheduler!=None and no_improvement>=5:
                    no_improvement=0
                    scheduler.step()

                if ep%5==0:
                    sched_path = f"scheduler_{ep}.pth"
                    opt_path = f"optimizer_{ep}.pth"
                    model_path = f"model_{ep}.pth"
                    interface_path = f"interface_{ep}.pth"

                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, model_path))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, opt_path))
                    torch.save(interface.state_dict(), os.path.join(self.config.runner.ckpt_path, interface_path))
                    if scheduler:
                        torch.save(scheduler.state_dict(), os.path.join(self.config.runner.ckpt_path, sched_path))
                ep+=1

    def inference(self, test_loader, model, device, parallel=False, out_dir='./'):

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
        
        #Load checkpoint
        ckpt_path = os.path.join(self.config.runner.ckpt_path, self.config.runner.restore_epoch)
        model.load_state_dict(torch.load(ckpt_path))

        
        if self.config.model.model_name=='speaker_encoder':
            save_output = None
            save_spk = None
            
            print("Generating test data...")
            for i, batch in enumerate(test_loader):
                if i%200==0:
                    print(f"> [Step:]{i + 1}/{len(test_loader)}")

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

                if i==0:
                    save_output = outputs
                    save_spk = spk_true
                else:
                    save_output = np.concatenate([save_output, outputs], axis=0)
                    save_spk = np.concatenate([save_spk, spk_true], axis=0)

            np.save(os.path.join(out_dir, 'X.npy'), save_output)
            np.save(os.path.join(out_dir, 'Y.npy'), save_spk)

            print(f'Saved inference results at {out_dir}')
        
        elif self.config.model.model_name=='joint_vc':
            interface = Interface()
            if self.config.model.interface:
                interface.load_state_dict(torch.load(self.config.model.interface))

            feat_extractor = FeatureExtractor(model=model, layers=["speaker_encoder", "attr_predictor"])
            for i, batch in enumerate(test_loader):
                x = batch['x'].to(device)
                text = batch['text'].to(device)
                target = batch["target"].to(device)
                spk_true = batch['spk_id'].to(device)
                d_labels = batch['d_labels'].to(device)
                outs = model(x, text, target, interface)
                features = feat_extractor(x, text, target, interface)
                print({name: output.shape for name, output in features.items()})