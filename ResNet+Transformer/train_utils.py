import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))


crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        x_ = x.contiguous().view(-1, x.size(-1))
        y_ = y.contiguous().view(-1)
        loss = self.criterion(x_, y_)
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
       
        return loss.item() * norm

def run_epoch(data_loader, model, loss_compute, device=None):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)
        decode_in = decode_in.to(device)
        decode_out = decode_out.to(device)
        decode_mask = decode_mask.to(device)
        ntokens = torch.sum(ntokens).to(device)

        out = model.forward(img_input, decode_in, encode_mask, decode_mask)
        # out --> [bs, 20, 512]  预测结果
        # decode_out --> [bs, 20]  实际结果
        # ntokens --> 标签中实际有效字符

        loss = loss_compute(out, decode_out, ntokens)  # 损失计算
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 100 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(model,train_loader,valid_loader,criterion,model_opt,input_path,config):
    epochs = config['epochs']
    device = config['device']
    verbose = config['verbose']
    weights_save_path = input_path['weights_save_path']
    history = {'train_loss':[],'valid_loss':[],'lr':[],'best_val_loss':1e5,'best_epoch':0}
    for epoch in range(1,epochs+1):

        t1 = time.time()
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
        train_mean_loss = run_epoch(train_loader, model, loss_compute, device)
        history['train_loss'].append(train_mean_loss)

        if epoch % verbose == 0:
            model.eval()
            valid_loss_compute = SimpleLossCompute(model.generator, criterion, None)
            valid_mean_loss = run_epoch(valid_loader, model, valid_loss_compute, device)
            history['valid_loss'].append(valid_mean_loss)
            print(f"EPOCH[{epoch}]: valid loss: {valid_mean_loss}",end='\n\n')

            learning_rate = model_opt.optimizer.state_dict()['param_groups'][0]['lr']
            history['lr'].append(learning_rate)

            # save model
            if valid_mean_loss < history['best_val_loss']:
                history['best_val_loss'] = valid_mean_loss
                history['best_epoch'] = epoch
                torch.save(model.state_dict(), weights_save_path, pickle_protocol=3)

        t2 = time.time()
        print('--------------耗时计算-------------------')
        epoch_time = (t2-t1)/60.0 # min
        print('Time of the last epoch: %.3f min'%epoch_time)
        print('-----------------------------------------')

    model.load_state_dict(torch.load(weights_save_path))
    np.save('history.npy',history)
    return model, history


