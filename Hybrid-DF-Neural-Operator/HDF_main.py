"""
predict AEM response Breal and Bimag
usage: python HDF_main.py HDF_config
@author: Zhiyuan Ke
"""
import os
import numpy as np
import torch
from torchinfo import summary
import yaml
import sys
from timeit import default_timer
from utilities import *
from HDF_3D import HDF

torch.manual_seed(0)
np.random.seed(0)

def print_model(model, flag=True):
    if flag:
        summary(model)

def get_batch_data(TRAIN_PATH1,TRAIN_PATH2,TRAIN_PATH3,TRAIN_PATH4,TRAIN_PATH5,TRAIN_PATH6,TRAIN_PATH7,TRAIN_PATH8,TRAIN_PATH9, TEST_PATH1, ntrain, ntest,  s_train,  s_test, batch_size, n_out):

    print("===============================")
    print("Start to read all training data")
    print("===============================")
    key_map0 = ['BZreal','BZimag'] # we only use Breal and Bimag
    key_map = key_map0[:n_out] # number of output channels is 2 (Breal, Bimag)
    t_read0 = default_timer()
    # get training data
    reader1 = MatReader(TRAIN_PATH1) # class in utilities
    x_train1 = reader1.read_field('sig')
    x_train1 = x_train1[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader2 = MatReader(TRAIN_PATH2)  # class in utilities
    x_train2 = reader2.read_field('sig')
    x_train2 = x_train2[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader3 = MatReader(TRAIN_PATH3)  # class in utilities
    x_train3 = reader3.read_field('sig')
    x_train3 = x_train3[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader4 = MatReader(TRAIN_PATH4)  # class in utilities
    x_train4 = reader4.read_field('sig')
    x_train4 = x_train4[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader5 = MatReader(TRAIN_PATH5)  # class in utilities
    x_train5 = reader5.read_field('sig')
    x_train5 = x_train5[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader6 = MatReader(TRAIN_PATH6)  # class in utilities
    x_train6 = reader6.read_field('sig')
    x_train6 = x_train6[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader7 = MatReader(TRAIN_PATH7)  # class in utilities
    x_train7 = reader7.read_field('sig')
    x_train7 = x_train7[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader8 = MatReader(TRAIN_PATH8)  # class in utilities
    x_train8 = reader8.read_field('sig')
    x_train8 = x_train8[:3000, :s_train[0], :s_train[1], :s_train[2]]
    reader9 = MatReader(TRAIN_PATH9)  # class in utilities
    x_train9 = reader9.read_field('sig')
    x_train9 = x_train9[:3000, :s_train[0], :s_train[1], :s_train[2]]
    x_train_1 = np.concatenate((x_train1,x_train2,x_train3))
    x_train_2 = np.concatenate( (x_train4, x_train5, x_train6))
    x_train_3 = np.concatenate((x_train7, x_train8, x_train9))
    x_train = np.concatenate((x_train_1, x_train_2, x_train_3))
    x_train =x_train[:,:,:,:20]  # we only use the former 20 layers in our fwd process
    x_train = np.abs(x_train)  # change the conductivity to resistivity
    print('x_train.shape=', x_train.shape)
    y_train1 = torch.stack([reader1.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    print('y_train1.shape=', y_train1.shape)
    y_train2 = torch.stack([reader2.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train3 = torch.stack([reader3.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train4 = torch.stack([reader4.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train5 = torch.stack([reader5.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train6 = torch.stack([reader6.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]]for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train7 = torch.stack([reader7.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train8 = torch.stack([reader8.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train9 = torch.stack([reader9.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
    y_train_1 = np.concatenate((y_train1, y_train2, y_train3))
    y_train_2 = np.concatenate((y_train4, y_train5, y_train6))
    y_train_3 = np.concatenate((y_train7, y_train8, y_train9))
    y_train = np.concatenate((y_train_1, y_train_2, y_train_3))
    print('y_train.shape=', y_train.shape)

    freq_base    = reader1.read_field('freq')[0]
    obs_base     = reader1.read_field('obs')[0]
    freq    = torch.log10(freq_base[:s_train[3]]) # frequency normalization
    obs     = obs_base[:s_train[4]]/torch.max(obs_base) # receivers' location normalization
    nLoc = obs.shape[0]
    nFreq = freq.shape[0]
    freq = freq.view(nFreq, -1)

    del reader1 # delete the reader object to save memory
    del reader2  # delete the reader object to save memory
    del reader3  # delete the reader object to save memory
    del reader4  # delete the reader object to save memory
    del reader5  # delete the reader object to save memory
    del reader6  # delete the reader object to save memory
    del reader7  # delete the reader object to save memory
    del reader8  # delete the reader object to save memory
    del reader9  # delete the reader object to save memory

    # get test data
    reader1_test = MatReader(TEST_PATH1)
    x_test1 = reader1_test.read_field('sig')
    x_test1 = x_test1[:3000,:s_test[0],:s_test[1],:s_test[2]]
    x_test = x_test1[:, :, :, :20]  # we only use the former 20 layers in our fwd process
    x_test = np.abs(x_test)  # change the conductivity to resistivity
    print('x_test.shape=', x_test.shape)
    y_test1 = torch.stack([reader1_test.read_field(key_map[i])[:3000,:s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1,3,2,0)
    y_test = y_test1
    print('y_test.shape=',y_test.shape)
    freq = torch.log10(freq_base[:s_test[3]])
    obs = obs_base[:s_test[4]] / torch.max(obs_base)
    nLoc = obs.shape[0]
    nFreq = freq.shape[0]
    freq = freq.view(nFreq, -1)

    del reader1_test

    # dataset normalization
    x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test)
    x_normalizer = GaussianNormalizer(x_train) 
    x_train = x_normalizer.encode(x_train) 
    x_test = x_normalizer.encode(x_test)

    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    y_normalizer = GaussianNormalizer_out(y_train) 
    y_train = y_normalizer.encode(y_train) 

    x_train = x_train.reshape(ntrain,s_train[0],s_train[1],s_train[2],1)
    x_test = x_test.reshape(ntest,s_test[0],s_test[1],s_test[2],1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    t_read1 = default_timer()
    print("===============================")
    print(f"all data has been read in {t_read1-t_read0:.3f} senconds")
    print("===============================")

    return train_loader, test_loader, freq, nLoc, x_normalizer, y_normalizer

def batch_train(model, freq, train_loader, y_normalizer, loss_func, optimizer, scheduler, device):

    train_l2 = 0.0
    freq = freq.to(device)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, freq)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()        
        train_l2 += loss.item()
    scheduler.step()
    return train_l2


def batch_validate(model, freq, test_loader, y_normalizer, loss_func, device):

    test_l2 = 0.0
    freq = freq.to(device)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, freq)
            out = y_normalizer.decode(out)
            test_l2 += loss_func(out, y).item()
    return test_l2


def run_train(model, freq, train_loader, test_loader,
              y_normalizer, loss_func, optimizer, scheduler, 
              epochs, thre_epoch, patience, save_step, 
              save_mode, model_path, model_path_temp,
              ntrain, ntest, device, log_file):
    
    val_l2 = np.inf
    stop_counter = 0

    temp_file = None
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_l2 = batch_train(model, freq, train_loader, y_normalizer, loss_func, optimizer, scheduler, device)
        model.eval()
        test_l2 = batch_validate(model, freq, test_loader, y_normalizer, loss_func, device)
        train_l2/= ntrain
        test_l2 /= ntest

        # save model
        if (ep+1) % save_step == 0:
            if temp_file is not None:
                os.remove(temp_file)
            torch.save(model.state_dict(), model_path_temp + '_epoch_' + str(ep+1) + '.pkl')
            temp_file = model_path_temp + '_epoch_' + str(ep + 1) + '.pkl'


        if (ep+1) > thre_epoch:
            if test_l2 < val_l2:
                val_l2 = test_l2
                stop_counter = 0 
                if save_mode == 'state_dict':
                    torch.save(model.state_dict(), model_path + '_epoch_' + str(thre_epoch) + '.pkl')
                else:
                    torch.save(model, model_path + '_epoch_' + str(thre_epoch) + '.pt')
            else:
                stop_counter += 1
            if stop_counter > patience: 
                print(f"Early stop at epoch {ep}")
                print(f"# Early stop at epoch {ep}",file=log_file)
                break

        t2 = default_timer()
        print(ep + 1, t2 - t1, train_l2, test_l2)
        print(ep + 1, t2 - t1, train_l2, test_l2, file = log_file)


# main function
def main(item):

    t0 = default_timer()
    with open( 'config_HDF.yml') as f:
        config = yaml.full_load(f)
    config = config[item]
    cuda_id = "cuda:" + str(config['cuda_id'])
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    TRAIN_PATH1 = config['TRAIN_PATH1']
    TRAIN_PATH2 = config['TRAIN_PATH2']
    TRAIN_PATH3 = config['TRAIN_PATH3']
    TRAIN_PATH4 = config['TRAIN_PATH4']
    TRAIN_PATH5 = config['TRAIN_PATH5']
    TRAIN_PATH6 = config['TRAIN_PATH6']
    TRAIN_PATH7 = config['TRAIN_PATH7']
    TRAIN_PATH8 = config['TRAIN_PATH8']
    TRAIN_PATH9 = config['TRAIN_PATH9']
    TEST_PATH1  = config['TEST_PATH1']
    save_mode  = config['save_mode']
    save_step  = config['save_step']
    n_out      = config['n_out']
    model_path = "model/" + config['name'] + "_" + str(n_out)
    model_path_temp = "temp_model/" + config['name'] + "_" + str(n_out)
    log_path = "log/" + config['name'] + "_" + str(n_out) + '.log'
    ntrain = config['ntrain']
    ntest  = config['ntest']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    step_size = config['step_size']
    gamma = config['gamma']
    modes = config['modes']
    width = config['width']
    s_train = config['s_train']
    s_test = config['s_test']
    layer_fno = config['layer_fno']
    layer_ufno = config['layer_ufno']
    layer_sizes = [s_train[0] * s_train[1]* s_train[2]] + config['layer_sizes']
    act_fno   = config['act_fno']
    init_func = config['init_func']    
    patience = config['patience']
    thre_epoch = config['thre_epoch']
    print_model_flag = config['print_model_flag']

    # load data and data normalization 
    train_loader, test_loader, freq, nLoc, _, y_normalizer = \
    get_batch_data(TRAIN_PATH1,TRAIN_PATH2,TRAIN_PATH3,TRAIN_PATH4,TRAIN_PATH5,TRAIN_PATH6,TRAIN_PATH7,TRAIN_PATH8,TRAIN_PATH9, TEST_PATH1, ntrain, ntest, \
                    s_train,s_test,batch_size,n_out)
    y_normalizer.to(device)

    model = HDF(modes, modes,modes, width, n_out, layer_sizes, nLoc, init_func, layer_fno, layer_ufno, act_fno).to(device)

    print_model(model, print_model_flag)

    # setup the optimizer, learning decay, loss function
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = LpLoss(size_average=False)

    # network training starts, and save the training log file
    log_file = open(log_path,'a+')
    print("===============================")
    print("Now we're begin to train the HDF")
    print("===============================")
    run_train(model, freq, train_loader, test_loader, y_normalizer, myloss, \
              optimizer, scheduler, epochs, thre_epoch, patience, save_step, \
              save_mode, model_path, model_path_temp, ntrain, ntest, device, log_file)
    tn = default_timer()
    print("===============================")
    print(f'all time:{tn-t0:.3f}s')
    print(f'# all time:{tn-t0:.3f}s',file=log_file)
    print("===============================")
    log_file.close()


if __name__ == '__main__':
    try:
        item = sys.argv[1]
    except:
        item = 'HDF_config'
    main(item)