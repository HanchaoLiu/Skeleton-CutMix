import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from config import cfg 

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

import glob

from baseloss import loss_ccsa, loss_dage, loss_dsne

###############################################################################################
# important!
# st-gcn has model.apply(weight_init)
# agcn does not have weight init
# 
###############################################################################################

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')

    parser.add_argument('--tensorboard', type=str2bool, default=False, help='use tensorboard or not')
    # parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score',type=str2bool,default=False,help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=0, help='random seed for pytorch')
    parser.add_argument('--log-interval',type=int,default=100,help='the interval for printing messages (#iteration)')

    parser.add_argument('--save-interval',type=int,default=2,help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval',type=int,default=5,help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    parser.add_argument('--show-topk',type=int,default=[1, 5],nargs='+',help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--train-feeder',default='feeder.feeder',help='data loader will be used')
    parser.add_argument('--train-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for training')
    parser.add_argument('--num-worker',type=int,default=1,help='the number of worker for data loader')
    parser.add_argument('--test-feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--test-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for test')

    parser.add_argument('--use-val', type=str2bool,default=False, help='data loader will be used')
    parser.add_argument('--val-feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--val-feeder-args',action=DictAction,default=dict(),help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',action=DictAction,default=dict(),help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')
    parser.add_argument('--ignore-weights',type=str,default=[],nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
    parser.add_argument('--num-epoch',type=int,default=80,help='stop training in which epoch')
    parser.add_argument('--weight-decay',type=float,default=0.0005,help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)

    parser.add_argument('--use-aug', type=str2bool,default=False, help='data loader will be used')
    

    parser.add_argument('--weight_contrastive',type=float,default=0.1,help='weight decay for optimizer')
    parser.add_argument('--weight_dst',type=float,default=0.2,help='weight decay for optimizer')
    parser.add_argument('--margin',type=float,default=1.0,help='weight decay for optimizer')
    parser.add_argument('--loss_type',type=str)

    return parser



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.tensorboard:
            if arg.phase == 'train':

                log_dir=os.path.join(self.arg.save_dir, "summary")
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
            
                self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(log_dir, 'val'), 'val')
                self.val_writer = SummaryWriter(os.path.join(log_dir, 'test'), 'test')
                
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder_train = import_class(self.arg.train_feeder)
        print(Feeder_train)
        Feeder_val = import_class(self.arg.val_feeder)
        Feeder_test = import_class(self.arg.test_feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder_train(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

            if self.arg.use_val:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder_val(**self.arg.val_feeder_args),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    drop_last=False,
                    worker_init_fn=init_seed)
            
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder_test(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)


        # weight init for st-gcn
        # if self.arg.model.startswith("nets.stgcn"):
        # also apply weight init for agcn
        # self.model.apply(weights_init)
        if self.arg.model.startswith("nets.stgcn"):
            self.model.apply(weights_init)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            # localtime = time.asctime(time.localtime(time.time()))
            # str = "[ " + localtime + ' ] ' + str
            str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.print_log('-'*70)
        self.model.train()
        
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        lr = self.optimizer.param_groups[0]['lr']
        self.print_log('Training epoch: {}, lr = {}'.format(epoch, lr))

        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        if self.arg.tensorboard:
            self.train_writer.add_scalar('epoch', epoch, self.global_step)

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        # for batch_idx, (data, label, index) in enumerate(process):
        loss_src_list, loss_dst_list, loss_aug_list = [], [], []
        for batch_idx, (data_src, data_dst, label_src, label_dst, index) in enumerate(process):
            self.global_step += 1
            # get data
            data_src = Variable(data_src.float().cuda(self.output_device), requires_grad=False)
            data_dst = Variable(data_dst.float().cuda(self.output_device), requires_grad=False)
    
            label_src = Variable(label_src.long().cuda(self.output_device), requires_grad=False)
            label_dst = Variable(label_dst.long().cuda(self.output_device), requires_grad=False)
            label = label_src
            timer['dataloader'] += self.split_time()

            weight_dst = self.arg.weight_dst
            weight_contrastive = self.arg.weight_contrastive

            # forward
            output_src, feat_src = self.model(data_src, return_feat = True)
            loss_src = self.loss(output_src, label_src)

            if weight_dst!=0:
                output_dst, feat_dst = self.model(data_dst, return_feat = True)
                loss_dst = self.loss(output_dst, label_dst)
            else:
                loss_dst = torch.tensor(0.0).cuda()

            if self.arg.loss_type == 'ccsa':
                loss_aug = loss_ccsa(feat_src, label_src, feat_dst, label_dst, margin=self.arg.margin)
            elif self.arg.loss_type == 'dsne':
                loss_aug = loss_dsne(feat_src, label_src, feat_dst, label_dst, margin=self.arg.margin)
            elif self.arg.loss_type == 'dage':
                loss_aug = loss_dage(feat_src, label_src, feat_dst, label_dst)
            loss_aug = loss_aug.mean()
            
            loss_src_list.append(loss_src.item())
            loss_dst_list.append(loss_dst.item())
            loss_aug_list.append(loss_aug.item() )
            
            loss = loss_src + loss_dst * weight_dst + loss_aug * weight_contrastive
            
            
                
                

            

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output_src.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            if self.arg.tensorboard:
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            if self.arg.tensorboard:
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        self.print_log("loss_src={:.4f}, loss_dst={:.4f}, loss_aug={:.4f}".format(
            np.mean(loss_src_list), np.mean(loss_dst_list), np.mean(loss_aug_list)
        ))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')
            save_model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(epoch))
            torch.save(weights, save_model_name)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            gt_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    # data = Variable(
                    #     data.float().cuda(self.output_device),
                    #     requires_grad=False,
                    #     volatile=True)
                    # label = Variable(
                    #     label.long().cuda(self.output_device),
                    #     requires_grad=False,
                    #     volatile=True)

                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    gt_frag.append(label.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            pred  = np.argmax(score, axis=1)
            gt    = np.concatenate(gt_frag)


            loss = np.mean(loss_value)
            # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            accuracy = (pred==gt).mean()
            # if accuracy > self.best_acc:
            #     self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            self.result_dt[ln].append(accuracy)

            self.print_log('{} acccuracy = {:.4f}'.format(ln, accuracy) )
            if self.arg.phase == 'train':
                if self.arg.tensorboard:
                    self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                    self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:.5f}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            # for k in self.arg.show_topk:
            #     self.print_log('\tTop{}: {:.2f}%'.format(
            #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)


    def show_acc_res(self, save_res=True, show_file_name=False):
        # self.print_log(f"{self.result_dt['val']}")
        # self.print_log(f"{self.result_dt['test']}")

        test_acc_list = np.array(self.result_dt['test'])
        best_test_acc = np.max(test_acc_list)
        best_test_idx = np.argmax(test_acc_list)

        if self.arg.use_val:
            val_acc_list = np.array(self.result_dt['val'])
            best_val_acc = np.max(val_acc_list)
            best_val_idx = np.argmax(val_acc_list)

        if self.arg.use_val:
            best_test_acc_at_val = test_acc_list[best_val_idx]
            # result_str = "best_val_acc={:.3f} @ ep{}, best_test_acc={:.3f} @ ep{}, best_test_acc@val={:.3f} @ ep{}".format(
            #     best_val_acc, best_val_idx,
            #     best_test_acc, best_test_idx,
            #     best_test_acc_at_val, best_val_idx
            # )
            result_str = "best_val_acc={:.3f} @ ep{}, best_test_acc={:.3f} @ ep{}".format(
                best_val_acc, best_val_idx,
                best_test_acc, best_test_idx
            )
        else:
            result_str = "best_test_acc={:.3f} @ep {}".format(
                best_test_acc, best_test_idx
            )
        self.print_log(result_str)

        if save_res:
            save_name_pkl = os.path.join(self.arg.base_dir, "result.pkl")
            with open(save_name_pkl, "wb") as f:
                pickle.dump(self.result_dt, f)
            
            

            save_name = os.path.join(self.arg.base_dir, "result.txt")
            result_str = result_str + ", last_test_acc = {:.3f}".format(test_acc_list[-1])
            with open(save_name, "w") as f:
                f.write(result_str+"\n")

            save_plot_name = os.path.join(self.arg.base_dir, "result.png")
            if self.arg.use_val:
                from vis.curve import plot_curve_test_val
                plot_curve_test_val(save_plot_name, self.result_dt['test'], self.result_dt['val'], self.arg.seed)
            else:
                from vis.curve import plot_curve_test
                plot_curve_test(save_plot_name, self.result_dt['test'], self.arg.seed)

            if show_file_name:
                self.print_log(f"result save to {save_name_pkl}")
                self.print_log(f"result save to {save_name}")
                self.print_log(f"log save to {self.print_file_name}")
                self.print_log(f"fig save to {save_plot_name}")


            # + save best model
            best_model_name = os.path.join(self.arg.work_dir, 'model_ep{}.pt'.format(best_test_idx))
            best_model_name_save = os.path.join(self.arg.work_dir, 'model_epbest.pt')
            shutil.copy(best_model_name, best_model_name_save)

            


    def start(self):

        # remove old log dir
        self.print_file_name = '{}/log.txt'.format(self.arg.work_dir)
        if os.path.exists(self.print_file_name):
            os.remove(self.print_file_name)
        
        import json
        a=json.dumps(vars(self.arg), indent=2)
        self.print_log(a)
        self.print_log("work_dir = "+self.arg.work_dir)
        self.print_log("base_dir = "+self.arg.base_dir)

        self.result_dt = {'val': [], 'test': []}

        if self.arg.phase == 'train':
            # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # if self.lr < 1e-3:
                #     break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    if self.arg.use_val:
                        self.eval( epoch, save_score=self.arg.save_score, loader_name=['val'])
                    self.eval( epoch, save_score=self.arg.save_score, loader_name=['test'])

                self.show_acc_res(save_res=True, show_file_name=False)

            self.show_acc_res(save_res=True, show_file_name=True)

            # + delete other models
            model_name_to_remove = glob.glob(os.path.join(self.arg.work_dir, "*.pt"))
            model_name_to_remove = [name for name in model_name_to_remove if 'best' not in name]
            for name in model_name_to_remove:
                os.remove(name)
            
            # + copy log file to result_dir
            old_log_file = '{}/log.txt'.format(self.arg.work_dir)
            new_log_file = '{}/log.txt'.format(self.arg.base_dir)
            shutil.copy(old_log_file, new_log_file)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    work_dir=arg.work_dir
    arg.work_dir = os.path.join(cfg.work_dir_base, work_dir)
    arg.base_dir = os.path.join(cfg.base_dir_base, work_dir)
    for k in arg.train_feeder_args.keys():
        if k.startswith('data_path') or k.startswith('label_path'):
            arg.train_feeder_args[k] = os.path.join( cfg.data_dir_base, arg.train_feeder_args[k] )
    # arg.train_feeder_args['data_path'] = os.path.join( cfg.data_dir_base, arg.train_feeder_args['data_path'] )
    # arg.train_feeder_args['label_path'] = os.path.join( cfg.data_dir_base, arg.train_feeder_args['label_path'] )

    if not os.path.exists(arg.work_dir):
        os.mkdir(arg.work_dir)
    if not os.path.exists(arg.base_dir):
        os.mkdir(arg.base_dir)

    if arg.use_val:
        arg.val_feeder_args['data_path'] = os.path.join( cfg.data_dir_base, arg.val_feeder_args['data_path'] )
        arg.val_feeder_args['label_path'] = os.path.join( cfg.data_dir_base, arg.val_feeder_args['label_path'] )

    arg.test_feeder_args['data_path'] = os.path.join( cfg.data_dir_base, arg.test_feeder_args['data_path'] )
    arg.test_feeder_args['label_path'] = os.path.join( cfg.data_dir_base, arg.test_feeder_args['label_path'] )

    processor = Processor(arg)
    processor.start()



if __name__ == '__main__':
    
    main()


