# basic package
import os
import yaml
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# pytorch package
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# predefined class
# from downstream_heliofm.HelioFM.datasets.helio import SolarFlareDataset
from downstream_heliofm.HelioFM.models.helio_spectformer import HelioSolarFlare
from .scripts.configs import get_args
from .scripts.model import Alexnet, Mobilenet, ResNet18, ResNet34, ResNet50
from .scripts.train import SolarFlSets, heliofm_FLDataset, HSS2, TSS, F1Pos, train_loop, test_loop, train_loop_spectformer, test_loop_spectformer

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print('1st check cuda..')
print('Number of available device', torch.cuda.device_count())
print('Current Device:', torch.cuda.current_device())
print('Device:', device)

# dataset partitions and create data frame
print('2nd process, loading data...')

# create parser here
# define arguments
args, config = get_args(
    config_dir="./baseline_fulldisk/scripts/configs/config_heliofm.yaml"
)

# print(train_dataset[0])
# print(train_dataset[0]['ts'].size())
# print(len(train_dataset))

# # define transformations / augmentation
# rotation = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomRotation(degrees=(-5,5)),
#     transforms.ToTensor()
# ])

# hr_flip = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=1.0),
#     transforms.ToTensor()
# ])

# vr_flip = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomVerticalFlip(p=1.0),
#     transforms.ToTensor()
# ])

# # define directory here
# img_dir = args.img_dir
# crr_dir = os.getcwd() + '/baseline_fulldisk/'

# # parameter search space
# weight_decay = [0, 1e-3, 1e-4]
# class_weight = [1.0, 3.0]

# print(f'Model: Spectformer')
# print(f'Hyper parameters: batch_size: {args.batch_size}, number of epoch: {args.epochs}')
# print(f'learning rate: {args.lr}, max learning rate: {args.max_lr}')
# print(f'class weight: {class_weight}, decay value: {weight_decay}')

data_train = heliofm_FLDataset(
        index_path=config.data.train_data_path,
        fl_path=config.data.train_data_path,
        time_delta_input_minutes=config.data.time_delta_input_minutes,
        time_delta_target_minutes=config.data.time_delta_target_minutes,
        n_input_timestamps=1,
        scalers=None,
        channels=config.data.channels,
    )

data_test = heliofm_FLDataset(
        index_path=config.data.valid_data_path,
        fl_path=config.data.valid_data_path,
        time_delta_input_minutes=config.data.time_delta_input_minutes,
        time_delta_target_minutes=config.data.time_delta_target_minutes,
        n_input_timestamps=1,
        scalers=None,
        channels=config.data.channels,
    )

# # Define dataset here! 
# train_file = f'24image_bin_class_train_12min.csv'
# test_file = f'24image_bin_class_test_12min.csv'

# # train set
# df_train = pd.read_csv(crr_dir + 'scripts/data/' + train_file)

# # test set and calibration set
# df_test = pd.read_csv(crr_dir + 'scripts/data/' + test_file)

# # string to datetime
# df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'], format = '%Y-%m-%d %H:%M:%S')
# df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

# # Define dataset
# # trainset, image augmentation
# positive_ins = df_train.loc[df_train['label']==1, :]
# negative_ins = df_train.loc[df_train['label']==0, :]
# df_pos = SolarFlSets(annotations_df = positive_ins, img_dir = img_dir, normalization = True)
# df_rotation = SolarFlSets(annotations_df = positive_ins, num_sample=8000, img_dir = img_dir, transform=rotation, normalization = True)
# df_vrflip = SolarFlSets(annotations_df = positive_ins, num_sample=8000, img_dir = img_dir, transform=vr_flip, normalization = True)
# df_hrflip = SolarFlSets(annotations_df = positive_ins, num_sample=8000, img_dir = img_dir, transform=hr_flip, normalization = True)
# df_over = SolarFlSets(annotations_df = positive_ins, num_sample=8000, img_dir = img_dir, normalization = True)

# df_neg = SolarFlSets(annotations_df = negative_ins, num_sample=None, img_dir = img_dir, normalization = True)
# # df_n_vrflip = SolarFlSets(annotations_df = negative_ins, num_sample=1000, img_dir = img_dir, transform=vr_flip, normalization = True)
# # df_n_rotation = SolarFlSets(annotations_df = negative_ins, num_sample=1000, img_dir = img_dir, transform=rotation, normalization = True)
# # df_n_hrflip = SolarFlSets(annotations_df = negative_ins, num_sample=1000, img_dir = img_dir, transform=hr_flip, normalization = True)
# # df_n_over = SolarFlSets(annotations_df = negative_ins, num_sample=1000, img_dir = img_dir, normalization = True)

# data_train = ConcatDataset([df_pos, df_rotation, df_vrflip, df_hrflip, df_over, df_neg]) #, df_n_rotation, df_n_vrflip, df_n_hrflip
# # testset
# data_test = SolarFlSets(annotations_df = df_test, img_dir = img_dir, normalization = True)

# num_pos = len(df_pos)+len(df_rotation)+len(df_vrflip)+len(df_hrflip)+len(df_over)
# num_neg = len(df_neg) #+len(df_n_rotation)+len(df_n_vrflip)+len(df_n_hrflip)
# print(f'positive samples: {num_pos}, negative samples: {num_neg}, imbalance ratio: {num_pos/num_neg:.2f}')

# Data loader
train_dataloader = DataLoader(data_train, batch_size = args.batch_size, shuffle = True) # num_workers = 0, pin_memory = True, 
test_dataloader = DataLoader(data_test, batch_size = args.batch_size, shuffle = False) # num_workers = 0, pin_memory = True,

# Cross-validatation with optimization ( total = 4folds X Learning rate sets X weight decay sets )
training_result = []
iter = 0
best_loss = float("inf") 
best_hsstss = 0
for wt in args.weight_decay:
    for cls_wt in args.class_weight:
    
        '''
        [ Grid search start here ] 
        - Be careful with  result array, model, loss, and optimizer
        - Their position matters
        '''
        # define model here
        # if args.models == 'Alexnet':
        #     net = Alexnet().to(device)
        # elif args.models == "Mobilenet":
        #     net = Mobilenet().to(device)
        # elif args.models == "Resnet18":
        #     net = ResNet18().to(device)
        # elif args.models == "Resnet34":
        #     net = ResNet34().to(device)
        # elif args.models == "Resnet50":
        #     net = ResNet50().to(device)
        # else:
        #     print("Model Selected: ", args.models)
        #     print('Invalid Model')
        #     exit()
        
        net = HelioSolarFlare(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_channels,
            embed_dim=config.model.embed_dim,
            time_embedding={"type": config.model.time_embedding.type,
                            "n_queries": config.model.time_embedding.n_queries,
                            "time_dim": config.model.time_embedding.time_dim},
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            drop_rate=config.model.drop_rate,
            dtype=config.dtype,
            window_size=config.model.window_size,
            dp_rank=config.model.dp_rank,
            learned_flow=config.model.learned_flow,
            use_latitude_in_learned_flow=config.use_latitude_in_learned_flow,
            init_weights=config.model.init_weights,
            checkpoint_layers=config.model.checkpoint_layers,
            n_spectral_blocks=config.model.spectral_blocks,
            rpe=config.model.rpe,
            finetune=config.finetune,
            nglo=config.nglo,
            config=config,
        ).to(device)
        

        # model setting
        model = nn.DataParallel(net, device_ids = [0]).to(device)

        # class weight
        device = next(model.parameters()).device
        class_weights = torch.tensor([1.0, cls_wt], dtype=torch.float).to(device)
        # loss_fn = nn.CrossEntropyLoss(weight = class_weights)
        loss_fn = nn.functional.binary_cross_entropy 
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = wt) 
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr = args.max_lr, # Upper learning rate boundaries in the cycle for each parameter group
                    steps_per_epoch = len(train_dataloader), # The number of steps per epoch to train for.
                    epochs = args.epochs, # The number of epochs to train for.
                    anneal_strategy = 'cos',
                    pct_start=0.7,
                    div_factor=1e5
                    )

        # initiate variable for finding best epoch
        iter += 1
        for t in range(args.epochs):
            
            # extract current time and compute training time
            t0 = time.time()
            datetime_object = datetime.datetime.fromtimestamp(t0)
            year = datetime_object.year
            month = datetime_object.month

            train_loss, train_result = train_loop_spectformer(train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=scheduler)
            test_loss, test_result = test_loop_spectformer(test_dataloader,  model=model, loss_fn=loss_fn)
            table = confusion_matrix(test_result[:, 1], test_result[:, 0]).ravel()
            HSS_score = HSS2(table)
            TSS_score = TSS(table)
            F1_score = f1_score(test_result[:, 1], test_result[:, 0], average='macro')
            
            # trace score and predictions
            duration = (time.time() - t0)/60
            actual_lr = optimizer.param_groups[0]['lr']
            training_result.append([t, actual_lr, wt, cls_wt, train_loss, test_loss, HSS_score, TSS_score, F1_score, duration])
            torch.cuda.empty_cache()

            # time consumption and report R-squared values.
            print(f'Epoch {t+1}: Lr: {actual_lr:.3e}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, HSS: {HSS_score:.4f}, TSS: {TSS_score:.4f}, F1: {F1_score:.4f}, Duration(min):  {duration:.2f}')

            check_hsstss = (HSS_score * TSS_score)**0.5
            if best_hsstss < check_hsstss:
                best_hsstss = check_hsstss
                best_loss = test_loss

                PATH = crr_dir + 'results/trained/' + f"{args.models}_{year}{month:02d}_train2011to2013_test2024_{args.filetag}_{iter}.pth"
            # save model
                torch.save({
                        'epoch': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training loss': train_loss,
                        'testing loss' : test_loss,
                        'HSS_test' : HSS_score,
                        'TSS_test' : TSS_score,
                        'F1_macro' : F1_score
                        }, PATH)
                
                # save prediction array
                pred_path = crr_dir + 'results/prediction/' + f'{args.models}_{year}{month:02d}_train2011to2013_test2024_{args.filetag}_{iter}.npy'
                
                with open(pred_path, 'wb') as f:
                    train_log = np.save(f, train_result)
                    test_log = np.save(f, test_result)

training_result.append([f'Hyper parameters: batch_size: {args.batch_size}, number of epoch: {args.epochs}, initial learning rate: {args.lr}'])

# save the results
#print("Saving the model's result")
df_result = pd.DataFrame(training_result, columns=['Epoch', 'learning rate', 'weight decay', 'class weight', 'Train_loss', 'Test_loss',
                                                    'HSS', 'TSS', 'F1_macro', 'Training-testing time(min)'])

total_save_path = crr_dir + 'results/validation/' + f'{args.models}_{year}{month:02d}_validation_{args.filetag}_results.csv'

print('Save file here:', total_save_path)
df_result.to_csv(total_save_path, index = False) 
        
print("Done!")