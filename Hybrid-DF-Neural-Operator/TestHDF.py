import sys
import os
import numpy as np
import torch
from torchinfo import summary
import yaml
import sys
from timeit import default_timer
from utilities import *
from HDF_3D import HDF

with open('config_HDF.yml') as f:
    config = yaml.full_load(f)
cuda_id = "cuda:"+"0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
item = 'HDF_config'
config = config[item]
TRAIN_PATH1 = config['TRAIN_PATH1']
TRAIN_PATH2 = config['TRAIN_PATH2']
TRAIN_PATH3 = config['TRAIN_PATH3']
TRAIN_PATH4 = config['TRAIN_PATH4']
TRAIN_PATH5 = config['TRAIN_PATH5']
TRAIN_PATH6 = config['TRAIN_PATH6']
TRAIN_PATH7 = config['TRAIN_PATH7']
TRAIN_PATH8 = config['TRAIN_PATH8']
TRAIN_PATH9 = config['TRAIN_PATH9']
TEST_PATH1 = config['TEST_PATH1']
save_mode = config['save_mode']
save_step = config['save_step']
n_out = config['n_out']  # bzreal, bzimag
model_path = "model/" + config['name'] + "_" + str(n_out) # save path and name of model
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
layer_sizes = [s_train[0] * s_train[1] * s_train[2]] + config['layer_sizes']
act_fno   = config['act_fno']
init_func = config['init_func']
patience = config['patience']
thre_epoch = config['thre_epoch']
print_model_flag = config['print_model_flag']

key_map0 = ['BZreal','BZimag'] # we only use Breal and Bimag
key_map = key_map0[:n_out]
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
x_train = x_train[:, :, :, :20]  # we only use the former 20 layers in our fwd process
x_train = np.abs(x_train)  # change conductivity to resistivity
print('x_train.shape=', x_train.shape)
y_train1 = torch.stack([reader1.read_field(key_map[i])[:3000, :s_train[4], :s_train[3]] for i in range(len(key_map))]).permute(1, 3, 2, 0)
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

freq_base = reader1.read_field('freq')[0]
obs_base = reader1.read_field('obs')[0]
freq    = torch.log10(freq_base[:s_train[3]]) # frequency normalization
obs     = obs_base[:s_train[4]]/torch.max(obs_base) # receivers' location normalization
nSample = x_train.shape[0]
nLoc = obs.shape[0]
nFreq = freq.shape[0]
freq = freq.view(nFreq, -1)

del reader1  # delete the reader object to save memory
del reader2  # delete the reader object to save memory
del reader3  # delete the reader object to save memory
del reader4  # delete the reader object to save memory
del reader5  # delete the reader object to save memory
del reader6  # delete the reader object to save memory
del reader7  # delete the reader object to save memory
del reader8  # delete the reader object to save memory
del reader9  # delete the reader object to save memory

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
nSample = x_test.shape[0]
nLoc = obs.shape[0]
nFreq = freq.shape[0]
freq = freq.view(nFreq, -1)

del reader1_test

model = HDF(modes, modes, modes, width, n_out, layer_sizes, nLoc, init_func, layer_fno, layer_ufno, act_fno).to(device)
model_path = sys.path[0] + "/temp_model/    .pkl"
print("model_path", model_path)
model.load_state_dict(torch.load(model_path))

# data normalization
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
x_normalizer = GaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
y_normalizer = GaussianNormalizer_out(y_train)
y_train = y_normalizer.encode(y_train)

def predict(x, freq=freq):

    x = x.unsqueeze(0).unsqueeze(4)
    x = x_normalizer.encode(x)
    freq = freq.to(device)
    x = x.to(device)
    with torch.no_grad():
        out = model(x,  freq)
        out = y_normalizer.decode(out)
    out = out.squeeze()

    return out.cpu().numpy()

from numpy import linalg as LA
reader_train1 = MatReader(TRAIN_PATH1)
x_train1 = reader_train1.read_field('sig')
y_train_BZreal1 = reader_train1.read_field('BZreal').numpy()
y_train_BZimag1 = reader_train1.read_field('BZimag').numpy()
reader_train2 = MatReader(TRAIN_PATH2)
x_train2 = reader_train2.read_field('sig')
y_train_BZreal2 = reader_train2.read_field('BZreal').numpy()
y_train_BZimag2 = reader_train2.read_field('BZimag').numpy()
reader_train3 = MatReader(TRAIN_PATH3)
x_train3 = reader_train3.read_field('sig')
y_train_BZreal3 = reader_train3.read_field('BZreal').numpy()
y_train_BZimag3 = reader_train3.read_field('BZimag').numpy()
reader_train4 = MatReader(TRAIN_PATH4)
x_train4 = reader_train4.read_field('sig')
y_train_BZreal4 = reader_train4.read_field('BZreal').numpy()
y_train_BZimag4 = reader_train4.read_field('BZimag').numpy()
reader_train5 = MatReader(TRAIN_PATH5)
x_train5 = reader_train5.read_field('sig')
y_train_BZreal5 = reader_train5.read_field('BZreal').numpy()
y_train_BZimag5 = reader_train5.read_field('BZimag').numpy()
reader_train6 = MatReader(TRAIN_PATH6)
x_train6 = reader_train6.read_field('sig')
y_train_BZreal6 = reader_train6.read_field('BZreal').numpy()
y_train_BZimag6 = reader_train6.read_field('BZimag').numpy()
reader_train7 = MatReader(TRAIN_PATH7)
x_train7 = reader_train7.read_field('sig')
y_train_BZreal7 = reader_train7.read_field('BZreal').numpy()
y_train_BZimag7 = reader_train7.read_field('BZimag').numpy()
reader_train8 = MatReader(TRAIN_PATH8)
x_train8 = reader_train8.read_field('sig')
y_train_BZreal8 = reader_train8.read_field('BZreal').numpy()
y_train_BZimag8 = reader_train8.read_field('BZimag').numpy()
reader_train9 = MatReader(TRAIN_PATH9)
x_train9 = reader_train9.read_field('sig')
y_train_BZreal9 = reader_train9.read_field('BZreal').numpy()
y_train_BZimag9= reader_train9.read_field('BZimag').numpy()
x_train_1 = np.concatenate((x_train1,x_train2,x_train3))
x_train_2 = np.concatenate( (x_train4, x_train5, x_train6))
x_train_3 = np.concatenate((x_train7, x_train8, x_train9))
x_train_final = np.concatenate((x_train_1, x_train_2, x_train_3))
x_train_final = x_train_final[:, :, :, :20]  # we only use the former 20 layers in our fwd process
x_train_final = np.abs(x_train_final)  # change conductivity to resistivity
y_train_BZreal_1 = np.concatenate((y_train_BZreal1, y_train_BZreal2, y_train_BZreal3))
y_train_BZreal_2 = np.concatenate((y_train_BZreal4, y_train_BZreal5, y_train_BZreal6))
y_train_BZreal_3 = np.concatenate((y_train_BZreal7, y_train_BZreal8, y_train_BZreal9))
y_train_BZreal = np.concatenate((y_train_BZreal_1, y_train_BZreal_2, y_train_BZreal_3))
y_train_BZimag_1 = np.concatenate((y_train_BZimag1, y_train_BZimag2, y_train_BZimag3))
y_train_BZimag_2 = np.concatenate((y_train_BZimag4, y_train_BZimag5, y_train_BZimag6))
y_train_BZimag_3 = np.concatenate((y_train_BZimag7, y_train_BZimag8, y_train_BZimag9))
y_train_BZimag = np.concatenate((y_train_BZimag_1, y_train_BZimag_2, y_train_BZimag_3))
x_train_final = torch.tensor(x_train_final)

nSample = x_train_final.shape[0]

err_all = np.zeros((nSample, 3))
for i in range(nSample):
    # print(i)
    xTrainNew = x_train_final[i, :, :, :]
    result = predict(xTrainNew)

    BZreal_predict = result[ :, :, 0]
    BZimag_predict = result[ :, :, 1]

    data1 = y_train_BZreal[i,  :, :]
    data2 = y_train_BZimag[i,  :, :]

    BZreal_predict = np.transpose(BZreal_predict)
    BZimag_predict = np.transpose(BZimag_predict)

    BZreal_predict_all = BZreal_predict.flatten()
    BZimag_predict_all = BZimag_predict.flatten()

    y_train_BZreal_all = data1.flatten()
    y_train_BZimag_all = data2.flatten()

    diff1 = (BZreal_predict_all - y_train_BZreal_all) / y_train_BZreal_all * 100
    err1 = LA.norm(diff1, 2) / len(y_train_BZreal_all) ** 0.5
    diff2 = (BZimag_predict_all - y_train_BZimag_all) / y_train_BZimag_all * 100
    err2 = LA.norm(diff2, 2) / len(y_train_BZimag_all) ** 0.5
    diff = np.power(diff1, 2) + np.power(diff2, 2)
    err = np.sqrt(np.sum(diff)) / (2 * len(diff)) ** 0.5
    err_all[i, 0] = err1
    err_all[i, 1] = err2
    err_all[i, 2] = err

# np.savetxt('y_train_BZreal_all.txt',y_train_BZreal_all)
# np.savetxt('y_train_BZimag_all.txt',y_train_BZimag_all)
# np.savetxt('BZreal_predict_all.txt',BZreal_predict_all)
# np.savetxt('BZimag_predict_all.txt',BZimag_predict_all)
err_data_new = err_all[:,2]
len(err_data_new[err_data_new>5].flatten())

nSample - len(err_data_new[err_data_new>5].flatten())
print("error < 5%: ", (1-len(err_data_new[err_data_new>5].flatten())/nSample)*100, "%")
nSample - len(err_data_new[err_data_new>10].flatten())
print("error < 10%: ", (1-len(err_data_new[err_data_new>10].flatten())/nSample)*100, "%")


import time
from numpy import linalg as LA
reader_train1 = MatReader(TRAIN_PATH1)
x_train1 = reader_train1.read_field('sig')
y_train_BZreal1 = reader_train1.read_field('BZreal').numpy()
y_train_BZimag1 = reader_train1.read_field('BZimag').numpy()
reader_train2 = MatReader(TRAIN_PATH2)
x_train2 = reader_train2.read_field('sig')
y_train_BZreal2 = reader_train2.read_field('BZreal').numpy()
y_train_BZimag2 = reader_train2.read_field('BZimag').numpy()
reader_train3 = MatReader(TRAIN_PATH3)
x_train3 = reader_train3.read_field('sig')
y_train_BZreal3 = reader_train3.read_field('BZreal').numpy()
y_train_BZimag3 = reader_train3.read_field('BZimag').numpy()
reader_train4 = MatReader(TRAIN_PATH4)
x_train4 = reader_train4.read_field('sig')
y_train_BZreal4 = reader_train4.read_field('BZreal').numpy()
y_train_BZimag4 = reader_train4.read_field('BZimag').numpy()
reader_train5 = MatReader(TRAIN_PATH5)
x_train5 = reader_train5.read_field('sig')
y_train_BZreal5 = reader_train5.read_field('BZreal').numpy()
y_train_BZimag5 = reader_train5.read_field('BZimag').numpy()
reader_train6 = MatReader(TRAIN_PATH6)
x_train6 = reader_train6.read_field('sig')
y_train_BZreal6 = reader_train6.read_field('BZreal').numpy()
y_train_BZimag6 = reader_train6.read_field('BZimag').numpy()
reader_train7 = MatReader(TRAIN_PATH7)
x_train7 = reader_train7.read_field('sig')
y_train_BZreal7 = reader_train7.read_field('BZreal').numpy()
y_train_BZimag7 = reader_train7.read_field('BZimag').numpy()
reader_train8 = MatReader(TRAIN_PATH8)
x_train8 = reader_train8.read_field('sig')
y_train_BZreal8 = reader_train8.read_field('BZreal').numpy()
y_train_BZimag8 = reader_train8.read_field('BZimag').numpy()
reader_train9 = MatReader(TRAIN_PATH9)
x_train9 = reader_train9.read_field('sig')
y_train_BZreal9 = reader_train9.read_field('BZreal').numpy()
y_train_BZimag9= reader_train9.read_field('BZimag').numpy()
x_train_1 = np.concatenate((x_train1, x_train2, x_train3))
x_train_2 = np.concatenate((x_train4, x_train5, x_train6))
x_train_3 = np.concatenate((x_train7, x_train8, x_train9))
x_train_final = np.concatenate((x_train_1, x_train_2, x_train_3))
x_train_final = x_train_final[:,:,:,:20]
x_train_final = np.abs(x_train_final)  # change conductivity to resistivity
y_train_BZreal_1 = np.concatenate((y_train_BZreal1, y_train_BZreal2, y_train_BZreal3))
y_train_BZreal_2 = np.concatenate((y_train_BZreal4, y_train_BZreal5, y_train_BZreal6))
y_train_BZreal_3 = np.concatenate((y_train_BZreal7, y_train_BZreal8, y_train_BZreal9))
y_train_BZreal = np.concatenate((y_train_BZreal_1, y_train_BZreal_2, y_train_BZreal_3))
y_train_BZimag_1 = np.concatenate((y_train_BZimag1, y_train_BZimag2, y_train_BZimag3))
y_train_BZimag_2 = np.concatenate((y_train_BZimag4, y_train_BZimag5, y_train_BZimag6))
y_train_BZimag_3 = np.concatenate((y_train_BZimag7, y_train_BZimag8, y_train_BZimag9))
y_train_BZimag = np.concatenate((y_train_BZimag_1, y_train_BZimag_2, y_train_BZimag_3))
nSample = 1000
x_train_final = torch.tensor(x_train_final)
x_test = torch.tensor(x_test)
y_train_BZreal = torch.tensor(y_train_BZreal)
y_train_BZimag = torch.tensor(y_train_BZimag)
y_test = torch.tensor(y_test)

start_time = time.time() 
err_all = np.zeros((nSample, 3))
for i in range(nSample):
    xTrainNew = x_train_final[i,:,:,:]
    result = predict(xTrainNew)
end_time = time.time()    
run_time = end_time - start_time   
print("The run time of one single prediction is : %.06f seconds" %(run_time/nSample))


from numpy import linalg as LA
reader_train1 = MatReader(TRAIN_PATH1)
x_train1 = reader_train1.read_field('sig')
y_train_BZreal1 = reader_train1.read_field('BZreal').numpy()
y_train_BZimag1 = reader_train1.read_field('BZimag').numpy()
reader_train2 = MatReader(TRAIN_PATH2)
x_train2 = reader_train2.read_field('sig')
y_train_BZreal2 = reader_train2.read_field('BZreal').numpy()
y_train_BZimag2 = reader_train2.read_field('BZimag').numpy()
reader_train3 = MatReader(TRAIN_PATH3)
x_train3 = reader_train3.read_field('sig')
y_train_BZreal3 = reader_train3.read_field('BZreal').numpy()
y_train_BZimag3 = reader_train3.read_field('BZimag').numpy()
reader_train4 = MatReader(TRAIN_PATH4)
x_train4 = reader_train4.read_field('sig')
y_train_BZreal4 = reader_train4.read_field('BZreal').numpy()
y_train_BZimag4 = reader_train4.read_field('BZimag').numpy()
reader_train5 = MatReader(TRAIN_PATH5)
x_train5 = reader_train5.read_field('sig')
y_train_BZreal5 = reader_train5.read_field('BZreal').numpy()
y_train_BZimag5 = reader_train5.read_field('BZimag').numpy()
reader_train6 = MatReader(TRAIN_PATH6)
x_train6 = reader_train6.read_field('sig')
y_train_BZreal6 = reader_train6.read_field('BZreal').numpy()
y_train_BZimag6 = reader_train6.read_field('BZimag').numpy()
reader_train7 = MatReader(TRAIN_PATH7)
x_train7 = reader_train7.read_field('sig')
y_train_BZreal7 = reader_train7.read_field('BZreal').numpy()
y_train_BZimag7 = reader_train7.read_field('BZimag').numpy()
reader_train8 = MatReader(TRAIN_PATH8)
x_train8 = reader_train8.read_field('sig')
y_train_BZreal8 = reader_train8.read_field('BZreal').numpy()
y_train_BZimag8 = reader_train8.read_field('BZimag').numpy()
reader_train9 = MatReader(TRAIN_PATH9)
x_train9 = reader_train9.read_field('sig')
y_train_BZreal9 = reader_train9.read_field('BZreal').numpy()
y_train_BZimag9= reader_train9.read_field('BZimag').numpy()
x_train_1 = np.concatenate((x_train1, x_train2, x_train3))
x_train_2 = np.concatenate((x_train4, x_train5, x_train6))
x_train_3 = np.concatenate((x_train7, x_train8, x_train9))
x_train_final = np.concatenate((x_train_1, x_train_2, x_train_3))
x_train_final = x_train_final[:,:,:,:20]
x_train_final = np.abs(x_train_final)  # change conductivity to resistivity
y_train_BZreal_1 = np.concatenate((y_train_BZreal1, y_train_BZreal2, y_train_BZreal3))
y_train_BZreal_2 = np.concatenate((y_train_BZreal4, y_train_BZreal5, y_train_BZreal6))
y_train_BZreal_3 = np.concatenate((y_train_BZreal7, y_train_BZreal8, y_train_BZreal9))
y_train_BZreal = np.concatenate((y_train_BZreal_1, y_train_BZreal_2, y_train_BZreal_3))
y_train_BZimag_1 = np.concatenate((y_train_BZimag1, y_train_BZimag2, y_train_BZimag3))
y_train_BZimag_2 = np.concatenate((y_train_BZimag4, y_train_BZimag5, y_train_BZimag6))
y_train_BZimag_3 = np.concatenate((y_train_BZimag7, y_train_BZimag8, y_train_BZimag9))
y_train_BZimag = np.concatenate((y_train_BZimag_1, y_train_BZimag_2, y_train_BZimag_3))
nSample = x_train_final.shape[0]

y_train_BZreal = y_train_BZreal.transpose(0, 2, 1)
y_train_BZimag = y_train_BZimag.transpose(0, 2, 1)
(nTol, nFreq, nSite) = y_train_BZreal.shape
print('y_train_BZreal.shape',y_train_BZreal.shape)
nnFreq = nFreq

err_all = np.zeros((nSample, 3))
predict_data1 = np.zeros((nSample,nnFreq, nSite,  2))
real_data1 = np.zeros((nSample,nnFreq, nSite,  2))
real_data1[:,:,:,0] = y_train_BZreal[:nSample,:,:]
real_data1[:,:,:,1] = y_train_BZimag[:nSample,:,:]
x_train_final = torch.tensor(x_train_final)

for i in range(nSample):

    xTrainNew = x_train_final[i, :, :, :]
    
    result = predict(xTrainNew)
    BZreal_predict = result[:,:,0]
    BZimag_predict = result[:,:,1]

    predict_data1[i,:,:,0] = BZreal_predict
    predict_data1[i,:,:,1] = BZimag_predict

real_data_reshape1 = real_data1.reshape(nSample, nnFreq, -1)
predict_data_reshape1 = predict_data1.reshape(nSample, nnFreq, -1)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 
r2_1 = np.zeros(nnFreq)
rmse_1 = np.zeros(nnFreq)
mae_1 = np.zeros(nnFreq)
r1 = real_data_reshape1
p1 = predict_data_reshape1

for i in range(nnFreq):
    r2_1[i] = r2_score(r1[:,i,:].flatten(), p1[:,i,:].flatten())
    rmse_1[i] = mean_squared_error(r1[:,i,:].flatten(), p1[:,i,:].flatten(), squared=False)
    mae_1[i] = mean_absolute_error(r1[:,i,:].flatten(), p1[:,i,:].flatten())
print("r2=================")
print(r2_1)
print("\nrmse=================")
print(rmse_1)
print("\nmae=================")
print(mae_1)

r2 = np.zeros(nFreq)
for i in range(nnFreq):
    r2[i] = r2_1[i]
rmse= np.zeros(nFreq)
for i in range(nnFreq):
    rmse[i] = rmse_1[i]
mae = np.zeros(nFreq)
for i in range(nnFreq):
    mae[i] = mae_1[i]
print(r2)
print(rmse)
print(mae)

def predict1(x, freq=freq1):

    x = x.unsqueeze(0).unsqueeze(4)
    x = x_normalizer.encode(x)
    x = x.to(device)
    freq = freq.to(device)
    with torch.no_grad():
        out = model(x,  freq)
        out = y_normalizer.decode(out)
    out = out.squeeze()

    return out.cpu().numpy()

from numpy import linalg as LA
reader_test1 = MatReader(TEST_PATH1)
x_test1 = reader_test1.read_field('sig')
y_test_BZreal1 = reader_test1.read_field('BZreal').numpy()
y_test_BZimag1 = reader_test1.read_field('BZimag').numpy()

x_test_final = x_test1
x_test_final = x_test_final[:, :, :, :20]  # we only use the former 20 layers in our fwd process
x_test_final = np.abs(x_test_final)  # change conductivity to resistivity
y_test_BZreal = y_test_BZreal1
y_test_BZimag = y_test_BZimag1

err_single = np.zeros((3))

# select sample
i = 1500
x_test_final = torch.tensor(x_test_final)
xtestNew = x_test_final[i,:,:,:]
result = predict(xtestNew)
BZreal_predict = result[:,:,0]
BZimag_predict = result[:,:,1]

data1 = y_test_BZreal[i,:,:]
data2 = y_test_BZimag[i,:,:]
data1 = data1
data2 = data2

BZreal_predict = np.transpose(BZreal_predict)
BZimag_predict = np.transpose(BZimag_predict)

BZreal_predict_all = BZreal_predict.flatten()
BZimag_predict_all = BZimag_predict.flatten()

y_test_BZreal_all = data1.flatten()
y_test_BZimag_all = data2.flatten()
    
diff1 = (BZreal_predict_all - y_test_BZreal_all)/y_test_BZreal_all*100
err1 = LA.norm(diff1, 2) / len(y_test_BZreal_all) ** 0.5
diff2 = (BZimag_predict_all - y_test_BZimag_all)/y_test_BZimag_all*100
err2 = LA.norm(diff2, 2) / len(y_test_BZimag_all) ** 0.5
diff = np.power(diff1, 2) + np.power(diff2, 2)
err = np.sqrt(np.sum(diff)) / (2*len(diff)) ** 0.5
err_single[0] = err1
err_single[1] = err2
err_single[2] = err
print(err_single)

from numpy import linalg as LA
reader_test1 = MatReader(TEST_PATH1)
x_test1 = reader_test1.read_field('sig')
y_test_BZreal1 = reader_test1.read_field('BZreal').numpy()
y_test_BZimag1 = reader_test1.read_field('BZimag').numpy()
x_test_final = x_test1
x_test_final = x_test_final[:, :, :, :20]  # we only use the former 20 layers in our fwd process
x_test_final = np.abs(x_test_final)  # change conductivity to resistivity
y_test_BZreal = y_test_BZreal1
y_test_BZimag = y_test_BZimag1
err_single = np.zeros((3))

i = 2500
x_test_final = torch.tensor(x_test_final)
xtestNew = x_test_final[i,:,:,:]
result = predict(xtestNew)
BZreal_predict = result[:,:,0]
BZimag_predict = result[:,:,1]

data1 = y_test_BZreal[i, :, :]
data2 = y_test_BZimag[i, :, :]
data1 = data1
data2 = data2

BZreal_predict = np.transpose(BZreal_predict)
BZimag_predict = np.transpose(BZimag_predict)

BZreal_predict_all = BZreal_predict.flatten()
BZimag_predict_all = BZimag_predict.flatten()

y_test_BZreal_all = data1.flatten()
y_test_BZimag_all = data2.flatten()

diff1 = (BZreal_predict_all - y_test_BZreal_all) / y_test_BZreal_all * 100
err1 = LA.norm(diff1, 2) / len(y_test_BZreal_all) ** 0.5
diff2 = (BZimag_predict_all - y_test_BZimag_all) / y_test_BZimag_all * 100
err2 = LA.norm(diff2, 2) / len(y_test_BZimag_all) ** 0.5
diff = np.power(diff1, 2) + np.power(diff2, 2)
err = np.sqrt(np.sum(diff)) / (2 * len(diff)) ** 0.5
err_single[0] = err1
err_single[1] = err2
err_single[2] = err
print(err_single)

from numpy import linalg as LA
reader_test1 = MatReader(TEST_PATH1)
x_test1 = reader_test1.read_field('sig')
y_test_BZreal1 = reader_test1.read_field('BZreal').numpy()
y_test_BZimag1 = reader_test1.read_field('BZimag').numpy()

x_test_final = x_test1
x_test_final = x_test_final[:, :, :, :20]  # we only use the former 20 layers in our fwd process
x_test_final = np.abs(x_test_final)  # change conductivity to resistivity
y_test_BZreal = y_test_BZreal1
y_test_BZimag = y_test_BZimag1
nSample = x_test_final.shape[0]

err_all = np.zeros((nSample, 3))
x_test_final = torch.tensor(x_test_final)
for i in range(nSample):
    xtestNew = x_test_final[i, :, :, :]
    result = predict(xtestNew)
    BZreal_predict = result[:, :, 0]
    BZimag_predict = result[:, :, 1]

    data1 = y_test_BZreal[i, :, :]
    data2 = y_test_BZimag[i, :, :]
    data1 = data1
    data2 = data2

    BZreal_predict = np.transpose(BZreal_predict)
    BZimag_predict = np.transpose(BZimag_predict)

    BZreal_predict_all = BZreal_predict.flatten()
    BZimag_predict_all = BZimag_predict.flatten()

    y_test_BZreal_all = data1.flatten()
    y_test_BZimag_all = data2.flatten()

    diff1 = (BZreal_predict_all - y_test_BZreal_all) / y_test_BZreal_all * 100
    err1 = LA.norm(diff1, 2) / len(y_test_BZreal_all) ** 0.5
    diff2 = (BZimag_predict_all - y_test_BZimag_all) / y_test_BZimag_all * 100
    err2 = LA.norm(diff2, 2) / len(y_test_BZimag_all) ** 0.5
    diff = np.power(diff1, 2) + np.power(diff2, 2)
    err = np.sqrt(np.sum(diff)) / (2 * len(diff)) ** 0.5
    err_all[i, 0] = err1
    err_all[i, 1] = err2
    err_all[i, 2] = err
    np.savetxt(f'results/y_test_BZreal/y_test_BZreal_all_{i + 1}.txt', y_test_BZreal_all)
    np.savetxt(f'results/y_test_BZimag/y_test_BZimag_all_{i + 1}.txt', y_test_BZimag_all)
    np.savetxt(f'results/BZreal_predict/BZreal_predict_all_{i + 1}.txt', BZreal_predict_all)
    np.savetxt(f'results/BZimag_predict/BZimag_predict_all_{i + 1}.txt', BZimag_predict_all)

err_data_new = err_all[:,2]
len(err_data_new[err_data_new>5].flatten())
nSample - len(err_data_new[err_data_new>5].flatten())
print("error < 5%: ", (1-len(err_data_new[err_data_new>5].flatten())/nSample)*100, "%")
nSample - len(err_data_new[err_data_new>10].flatten())
print("error < 10%: ", (1-len(err_data_new[err_data_new>10].flatten())/nSample)*100, "%")