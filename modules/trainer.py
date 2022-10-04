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
        #### General Encoder Training###
        ########################################

        if self.config.model.model_name=='general_encoder':
            #torch.set_num_threads(1)
            #print(f"Num threads: {torch.get_num_threads()}")
            if self.config.runner.restore_epoch:
                ep = self.config.runner.restore_epoch
                optimizer_to(optimizer, device)
            else:
                ep = 0
            step = 0
            prev_val_loss = 0
            no_improvement = 0

            if self.config.model.use_mi:
                mi_estimator = CLUB(self.config.mi_estimator.x_dim, 
                                    self.config.mi_estimator.y_dim,
                                    self.config.mi_estimator.hidden_dim).to(device)
                mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-4)
            else:
                mi_estimator = None
                mi_optimizer = None


            while ep < self.config.trainer.epoch:
                print(f"Starting [epoch]:{ep+1}/{self.config.trainer.epoch}")
                epoch_train_loss = 0
                epoch_train_loss_1 = 0
                epoch_train_loss_2 = 0
                epoch_mi_loss = 0
                epoch_mi_learn_loss = 0
                for i, batch in enumerate(train_loader):
                    if self.config.model.use_mi:
                        mi_estimator.eval()
                    x = batch['x'].to(device)
                    #p = batch['p'].to(device)
                    spk_true = batch['spk_id'].to(device)
                    optimizer.zero_grad()
                    outs = model(x)
                    loss, l1, l2, mi_loss = criterion(outs, spk_true, mi_estimator)
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    step+=1
                    epoch_train_loss+=loss.data
                    epoch_train_loss_1+=l1.data
                    epoch_train_loss_2+=l2.data
                    #epoch_mi_loss+=mi_loss.data

                    #MI estimator training

                    if self.config.model.use_mi:
                        for i in range(self.config.mi_estimator.mi_iter):
                            mi_estimator.train()
                            x = batch['x'].to(device)
                            outs = model(x)
                            x = outs["spk_emb"]
                            y =  outs["attr_emb"]
                            y = shuffle_tensor(y, dim=1)
                            mi_learn_loss = mi_estimator.learning_loss(x, y)
                            epoch_mi_learn_loss+=mi_loss.data
                            mi_learn_loss.backward()
                            mi_optimizer.step()
                        epoch_mi_learn_loss/= self.config.mi_estimator.mi_iter

                epoch_train_loss/=len(train_loader)
                epoch_train_loss_1/=len(train_loader)
                epoch_train_loss_2/=len(train_loader)
                    

                ##Validation loop
                val_loss = 0
                val_loss_1 = 0
                val_loss_2 = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['x'].to(device)
                        #p = batch['p'].to(device)
                        spk_true = batch['spk_id'].to(device)
                        outs = model(x)
                        loss, l1, l2, mi_loss = criterion(outs, spk_true, mi_estimator)
                        val_loss+=loss.data
                        val_loss_1+=l1.data
                        val_loss_2+=l2.data

                val_loss = val_loss/len(val_loader)
                val_loss_1 = val_loss_1/len(val_loader)
                val_loss_2 = val_loss_2/len(val_loader)

                print(f'> [Epoch]:{ep+1} [Train Loss]:{epoch_train_loss.data}')
                print(f'> [Epoch]:{ep+1} [Valid Loss]:{val_loss.data}')
                if self.config.runner.wandb:
                    wandb.log({"train_loss": epoch_train_loss.data,
                               "val_loss": val_loss.data, "l1_rc":epoch_train_loss_1,
                                "ce_loss":epoch_train_loss_2,
                                "mi_loss":epoch_mi_loss,
                                "val_l1_rc_loss":val_loss_1,
                                "val_ce_loss":val_loss_2})
                if val_loss < prev_val_loss:
                    #Save checkpoint and lr_sched state
                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, "best_model.pth"))
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

                    torch.save(model.state_dict(), os.path.join(self.config.runner.ckpt_path, model_path))
                    torch.save(optimizer.state_dict(), os.path.join(self.config.runner.ckpt_path, opt_path))
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