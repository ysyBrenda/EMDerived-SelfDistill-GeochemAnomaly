'''
This script handles the training process.
Including：
     Training of the teacher model
     Training of the stduent model (distillation)

author： ysyBrenda
run in env: torch >= 1.3.0
Date: 2025/03/30
'''
import argparse
import time
import dill as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Transformer,Transformer_s
from transformer.Optim import ScheduledOptim
import torch.utils.data as Data

# to use tensorboard，input following in terminal：
# $ tensorboard --logdir=output --port 6006
# if【was already in use】：lsof -i:6006， kill -9 PID

# Pay attention to the loss of training and val, if the model is overfitting, it cannot be used as a teacher model

def train_epoch(model, training_data, optimizer, opt, device):
    ''' Epoch operation in training'''
    model.train()
    total_loss = 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):  # todo
        # prepare data
        if opt.isContrastLoss:
            temp= batch[0].to(device)
            a = torch.chunk(temp, 3, dim=1)
            src_seq=torch.cat([a[0],a[1],a[2]],0)
        else:
            src_seq = batch[0].to(device)

        gold = batch[1][:, 2:].unsqueeze(1)
        trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])
        if opt.isContrastLoss:
             trg_seq=torch.cat([trg_seq,trg_seq,trg_seq],0)

        # forward
        optimizer.zero_grad()
        pred, *_ = model(src_seq, trg_seq)

        # backward and update parameters
        if opt.isContrastLoss:
            a = torch.chunk(pred, 3, dim=0)
            contras_loss = F.mse_loss(a[1].contiguous().view(-1), a[2].contiguous().view(-1), reduction='mean')
            loss = F.mse_loss(a[0].contiguous().view(-1), gold, reduction='mean') + opt.lambda_con * contras_loss
        else:
            if opt.loss==1:
                loss = F.l1_loss(pred.contiguous().view(-1), gold, reduction='mean')
            else:
                loss =F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')

        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()

    # print('total_train loss: {:8.5f},iter:{},average_train loss:{:8.5f} '.format(total_loss,len(training_data),total_loss/len(training_data))) #optimizer.n_steps=iter
    return total_loss/len(training_data)

def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation '''
    model.eval()
    total_loss = 0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            src_seq = batch[0].to(device)
            gold = batch[1][:, 2:].unsqueeze(1)
            trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])

            # forward
            pred, *_ = model(src_seq, trg_seq)

            # =============  loss==========================
            if opt.loss == 1:
                loss = F.l1_loss(pred.contiguous().view(-1), gold, reduction='mean')
            elif opt.loss ==2:
                loss = F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')

            total_loss += loss.item()
    # print('total_val loss:{:8.5f} ,iter:{},average_val loss:{:8.5f}'.format(total_loss,len(validation_data),total_loss/len(validation_data)))
    return total_loss/len(validation_data)


def train_epoch_student(model_t,model_s, training_data, optimizer, opt, device):
    ''' Epoch operation in training phase'''
    model_t.eval()
    model_s.train()
    total_loss = 0
    total_student_loss = 0
    total_kd_loss = 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):  # todo
        # prepare data
        src_seq = batch[0].to(device)

        gold = batch[1][:, 2:].unsqueeze(1)
        trg_seq, gold = map(lambda x: x.to(device), [batch[1].unsqueeze(1), gold.contiguous().view(-1)])
            #teacher:
        with torch.no_grad():
            pred_teacher, *_ =model_t(src_seq, trg_seq)

        # forward
        # model.train()
        optimizer.zero_grad()
        pred, *_ = model_s(src_seq, trg_seq)

        # =============  loss==========================
        if opt.loss == 1:
            student_loss = F.l1_loss(pred.contiguous().view(-1), gold, reduction='mean')
        elif opt.loss == 2:
            student_loss = F.mse_loss(pred.contiguous().view(-1), gold, reduction='mean')
        soft_loss= F.mse_loss(pred.contiguous().view(-1),pred_teacher.contiguous().view(-1), reduction='mean')
        alpha= opt.alpha # Influence of teachers
        loss=  (1-alpha) * student_loss +  alpha * soft_loss
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_student_loss += student_loss.item()
        total_kd_loss += soft_loss.item()
        model_s.train()

    # print('total_train loss: {:8.5f},iter:{},average_train loss:{:8.5f} '.format(total_loss,len(training_data),total_loss/len(training_data))) #optimizer.n_steps=iter
    return total_loss/len(training_data),model_s,total_student_loss/len(training_data),total_kd_loss/len(training_data)

def train(model, training_data, validation_data, optimizer, device, opt):
    """ Start training (teacher) """

    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from tensorboardX import SummaryWriter
        # from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard' + opt.fileHead))

    log_train_file = os.path.join(opt.output_dir, opt.fileHead, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, opt.fileHead, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,lr\n')
        log_vf.write('epoch,loss,lr\n')

    def print_performances(header, loss, start_time, lr):
        print('  - {header:12} loss: {loss: 8.5f},  lr: {lr: 8.2e}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", loss=loss,
            elapse=(time.time() - start_time) / 60, lr=lr))


    valid_losses = []
    LR=[]
    bad_counter = 0
    best = 1000000
    patience = 10  # 5
    be_teacher_epoch= 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(model, training_data, optimizer, opt, device)  # todo
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Teacher Training', train_loss, start, lr)

        start = time.time()
        valid_loss = eval_epoch(model, validation_data, device, opt)
        print_performances('Teacher Validation', valid_loss, start, lr)

        valid_losses += [valid_loss]
        LR+= [lr]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            # if epoch_i % 10 == 9:
                model_name = 'model_{epoch:d}_vloss_{vloss:.4f}.chkpt'.format(epoch=epoch_i, vloss=valid_loss)
                torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
        elif opt.save_mode == 'best':
            model_name = 'model_best.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
                print(' - [Info] The checkpoint file has been updated.')

        # ========== save model to "teacher model" file
        model_name = 'model_{TorS}.chkpt'.format(TorS=opt.TorS)
        teac_filehead = opt.fileHead.split('_')
        teac_filehead = '_'.join(teac_filehead[:-2])
        path = os.path.join(opt.output_dir, "teacher_model",teac_filehead,model_name)

        # ========== save model
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, path)
            be_teacher_epoch = epoch_i

        print(' - [Info] The best checkpoint file has been updated.')


        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=train_loss, lr=lr))
            log_vf.write('{epoch},{loss: 8.5f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=valid_loss, lr=lr))

        if opt.use_tb:
            tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

        # auto break
        if valid_loss < best:
            best = valid_loss
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == patience:
            break


    print(f"[Info] Model in epoch {be_teacher_epoch} will be teacher model.")
    # ------write a log---------------
    log_file_path = os.path.join(opt.output_dir, "teacher_model", teac_filehead, 'best_model_log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(
            f'{opt.TorS:<8}_Epoch {be_teacher_epoch}: Valid Loss = {valid_loss:8.5f}, lr_mul={opt.lr_mul:8.5f}\n')

    log_opt_file = 'opt_file_log.log'
    with open(log_opt_file, 'a') as log_f:
        log_f.write(str(opt.fileHead) + '__Teacher__loss__{:8.5f}\n'.format(valid_loss))

    return model

def train_student_KD(model_t,model_s, training_data, validation_data,optimizer, device, opt):
    """ Start training (student) """

    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard' + opt.fileHead))

    log_train_file = os.path.join(opt.output_dir, opt.fileHead, 'S_train.log')
    log_valid_file = os.path.join(opt.output_dir, opt.fileHead, 'S_valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,lr\n')
        log_vf.write('epoch,loss,lr\n')

    def print_performances(header, loss, start_time, lr):
        print('  - {header:12} loss: {loss: 8.5f},  lr: {lr: 8.2e}, ' \
              'elapse: {elapse:3.3f} min'.format(header=f"({header})", loss=loss,
            elapse=(time.time() - start_time) / 60, lr=lr))

    valid_losses = []
    LR=[]
    bad_counter = 0
    best = 100000
    patience = 10  # 5
    be_teacher_epoch= 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss,model_s,student_loss,kd_loss = train_epoch_student(model_t,model_s, training_data, optimizer, opt, device)
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Student Training', train_loss, start, lr)

        # start = time.time()
        valid_loss = eval_epoch(model_s, validation_data, device, opt)
        print_performances('Student Validation', valid_loss, start, lr)


        valid_losses += [valid_loss]
        LR += [lr]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model_s.state_dict()}
            
            #============save model to "teacher model" file
        model_name = 'model_{TorS}.chkpt'.format(TorS=opt.TorS)
        teac_filehead = opt.fileHead.split('_')
        teac_filehead = '_'.join(teac_filehead[:-2])
        path=os.path.join(opt.output_dir, "teacher_model",teac_filehead, model_name)
        # ============== save model
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, path)
            be_teacher_epoch = epoch_i

        print(' - [Info] The best checkpoint file has been updated.')

        
        if opt.save_mode == 'all':
            # if epoch_i % 10 == 9:
                model_name = 'model_{epoch:d}_vloss_{vloss:.4f}.chkpt'.format(epoch=epoch_i, vloss=valid_loss)
                torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
        elif opt.save_mode == 'best':
            model_name = 'model_best.chkpt'
            if be_teacher_epoch == epoch_i:
                torch.save(checkpoint, os.path.join(opt.output_dir, opt.fileHead, model_name))
                print(' - [Info] The checkpoint file has been updated.')


        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=train_loss, lr=lr))
            log_vf.write('{epoch},{loss: 8.9f},{lr:8.2e}\n'.format(
                epoch=epoch_i, loss=valid_loss, lr=lr))

        if opt.use_tb:
            tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


        # auto break

        if valid_loss < best:
            best = valid_loss
            bad_counter = 0
            print("bad_counter = 0")
        else:
            bad_counter += 1
            print(bad_counter )
        if bad_counter == patience:
            break


    log_opt_file = 'opt_file_log.log'
    with open(log_opt_file, 'a') as log_f:
        log_f.write(str(opt.fileHead) + '__Student__loss_{:.5f}\n'.format(valid_loss))

    print(f"[Info] Model in epoch {be_teacher_epoch} will be the next teacher model.")
    # ------write a log---------------
    log_file_path = os.path.join(opt.output_dir, "teacher_model", teac_filehead, 'best_model_log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{opt.TorS:<8}_Epoch {be_teacher_epoch}: Valid Loss = {valid_loss:8.5f}, lr_mul={opt.lr_mul:8.5f}\n')

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default='./data/pre_data.pkl')  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data
    parser.add_argument('-teacher_path', default='model_teacher.chkpt')  # 'model_Stud1.chkpt' ...

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=64)

    parser.add_argument('-d_model', type=int, default=38)  #todo
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=38)
    parser.add_argument('-d_v', type=int, default=38)

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=8)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)  # 0.1
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-output_dir', type=str, default='output')
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-unmask', type=float, default=0.5)
    parser.add_argument('-l2', type=float, default=0.0)  #  weight_dacay
    parser.add_argument('-lambda_con', type=float, default=0.0)  # contrast loss lambda  default=0.01
    parser.add_argument('-T', type=int, default=1)  # the times of mask
    parser.add_argument('-isContrastLoss', action='store_true')
    parser.add_argument('-isRandMask', action='store_true')
    parser.add_argument('-loss',type=int, default=2) #  loss fuction, l1_loss or l2_loss, input: 1 or 2
    parser.add_argument('-alpha', type=float, default=0.0)
    parser.add_argument('-TorS',  default='teacher')  # 'teacher','Stud1','Stud2'....

    opt = parser.parse_args()
    # # ++++++++++++++++
    opt.d_k = opt.d_model
    opt.d_v = opt.d_model

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model  # 512 ==>38

    # ------Output fileHead----
    opt.fileHead = 'T' + str(opt.T) + '_unmask' + str(opt.unmask) + '_h' + str(opt.n_head) + 'L' + str(
        opt.n_layers) + '_hid' + str(opt.d_inner_hid) + '_d'+ str(opt.d_model) + '_b' + str(
        opt.batch_size) + '_warm' + str(opt.n_warmup_steps) + '_seed' + \
                   str(opt.seed) + '_dr' + str(opt.dropout) +'_isCL'+str(opt.isContrastLoss)+ '_lamb'+str(
        opt.lambda_con) +'_ismask'+str(opt.isRandMask)+'_l'+str(opt.loss)+ '_alph'+str(opt.alpha)+ '_'+str(opt.TorS)
    if os.path.exists(os.path.join(opt.output_dir, opt.fileHead)):
        print('the output file is rewriting....', opt.fileHead)
    else:
        os.mkdir(os.path.join(opt.output_dir, opt.fileHead))
        print('The output filename is generated: ', opt.fileHead)

     #  teahcer model's file
    teac_filehead = opt.fileHead.split('_')
    teac_filehead = '_'.join(teac_filehead[:-2])
    
    # Check if teacher_model directory exists and create it if not
    if not os.path.exists(os.path.join(opt.output_dir, "teacher_model")):
        os.mkdir(os.path.join(opt.output_dir, "teacher_model"))
    if not os.path.exists(os.path.join(opt.output_dir, "teacher_model", teac_filehead)):
        os.mkdir(os.path.join(opt.output_dir, "teacher_model", teac_filehead))


    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # ========= Loading Dataset =========#
    training_data, validation_data = prepare_dataloaders(opt, device)
    print("training data size:{}, validation data size:{}".format(training_data.__len__(),validation_data.__len__()))

    
    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    print(opt)
    log_opt_file = os.path.join(opt.output_dir, opt.fileHead, 'opt.log')
    with open(log_opt_file, 'w') as log_f:
        log_f.write(str(opt))

    transformer_t = Transformer(
        src_pad_idx=opt.src_pad_idx,  # 1
        trg_pad_idx=opt.trg_pad_idx,  # 1
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer_t = ScheduledOptim(
        optim.Adam(transformer_t.parameters(), betas=(0.9, 0.98), eps=1e-09),  # ,weight_decay=opt.l2
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

   #======================Train teacher model // student model
    Teacher_Train=opt.TorS
    if Teacher_Train == 'teacher':
        transformer_t=train(transformer_t, training_data,validation_data, optimizer_t, device, opt)
        # for name, param in transformer_t.named_parameters():   #print parameters
        #     if name == "encoder.position_enc.linear_p.2.weight":
        #         print(name, param)
        #         break
        print('[Info] __ Teacher Model Has Trained ! __')
    else:
        print("opt.fileHead=    ", opt.fileHead)
        teac_filehead = opt.fileHead.split('_')
        teac_filehead = '_'.join(teac_filehead[:-2])
        print("teac_filehead=    ",teac_filehead)
        pretrained_chkpt =  os.path.join(opt.output_dir, "teacher_model",teac_filehead, opt.teacher_path )
        print(pretrained_chkpt)
        checkpoint = torch.load(pretrained_chkpt, map_location=device)
        transformer_t.load_state_dict(checkpoint['model'])
        print('[Info] __ Trained teacher model state loaded ! __',pretrained_chkpt)

        #  Student models
        transformer_s = Transformer_s(
            src_pad_idx=opt.src_pad_idx,  # 1
            trg_pad_idx=opt.trg_pad_idx,  # 1
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout).to(device)
        optimizer_s = ScheduledOptim(
            optim.Adam(transformer_s.parameters(), betas=(0.9, 0.98), eps=1e-09),  # ,weight_decay=opt.l2
            opt.lr_mul, opt.d_model, opt.n_warmup_steps)
        transformer_s.load_state_dict(checkpoint['model'])  # TODO
        train_student_KD(transformer_t, transformer_s, training_data, validation_data, optimizer_s, device, opt)


def Rand_mask(data, seed,unmask):
    data=data.clone()
    len = data.size(0)
    unmask = random.randint(int(len * unmask), len)  #number of mask ,randomly
    # unmask = int(len * 0.7)  # fix mask number
    random.seed(seed)
    torch.manual_seed(seed)

    shuffle_indices = torch.rand(len, len).argsort()
    unmask_ind, mask_ind = shuffle_indices[:, :unmask], shuffle_indices[:, unmask:]
    batch_ind = torch.arange(len).unsqueeze(-1)
    data[batch_ind, mask_ind] = 1  # padding=1
    return data



def prepare_dataloaders(opt, device):

    batch_size = opt.batch_size
    pkldata = pickle.load(open(opt.data_pkl, 'rb'))
    x = pkldata['x']
    y = pkldata['y']

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    x = torch.FloatTensor(np.array(x_train))
    y = torch.FloatTensor(np.array(y_train))
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))

    if opt.isRandMask:
        # x=train_torch_dataset.dataset
        print("~~~RandMask~~~~!")
        random.seed(opt.seed)  # NOTE: each generation seed can not be the same
        random_integers = [random.randint(0, 1000) for _ in range(10)]
        print("random_integers: ", random_integers)
        for T in range(0, opt.T):
            x1 = Rand_mask(x, random_integers[T], opt.unmask)
            if T == 0:
                if opt.isContrastLoss:
                    x1 = torch.cat([x1, Rand_mask(x, random_integers[T] + 3, opt.unmask),
                                    Rand_mask(x, random_integers[T] + 6, opt.unmask)], 1)  # data×3
                train_x = x1
                train_y = y
            else:
                if opt.isContrastLoss:
                    x1 = torch.cat([x1, Rand_mask(x, random_integers[T] + 3, opt.unmask),
                                    Rand_mask(x, random_integers[T] + 6, opt.unmask)], 1)
                train_x = torch.cat([train_x, x1], 0)
                train_y = torch.cat([train_y, y], 0)

            #add=========================
        random_file_name =os.path.join(opt.output_dir, opt.fileHead, "random.txt")
        with open(random_file_name, 'a') as f:
            for num in random_integers:
                f.write(f"{num} ")
    else:
        print("~~~No RandMask~~~~!")
        train_x = x
        train_y = y

    train_torch_dataset = Data.TensorDataset(train_x, train_y)
    val_torch_dataset = Data.TensorDataset(torch.FloatTensor(np.array(x_val)),torch.FloatTensor(np.array(y_val)))

    train_iterator = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # pin_memory=True
    )
    val_iterator = Data.DataLoader(
        dataset=val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    opt.src_pad_idx = 1
    opt.trg_pad_idx = 1

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()

    ''' 
    Usage:
    train teacher:
    python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS teacher

    train student:
    python train_KD.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 8 -warmup 128000 -lr_mul 200 -epoch 100 -b 16 -unmask 0.5 -T 1 -isRandMask -TorS Stud1 -teacher_path model_teacher.chkpt
    '''