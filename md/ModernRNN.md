# GRU

RNN存在的问题：梯度较容易出现衰减或爆炸（BPTT）  
⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系  
**RNN**:  


![Image Name](https://cdn.kesci.com/upload/image/q5jjvcykud.png?imageView2/0/w/320/h/320)


$$
H_{t} = ϕ(X_{t}W_{xh} + H_{t-1}W_{hh} + b_{h})
$$
**GRU**:


![Image Name](https://cdn.kesci.com/upload/image/q5jk0q9suq.png?imageView2/0/w/640/h/640)



$$
R_{t} = σ(X_tW_{xr} + H_{t−1}W_{hr} + b_r)\\    
Z_{t} = σ(X_tW_{xz} + H_{t−1}W_{hz} + b_z)\\  
\widetilde{H}_t = tanh(X_tW_{xh} + (R_t ⊙H_{t−1})W_{hh} + b_h)\\
H_t = Z_t⊙H_{t−1} + (1−Z_t)⊙\widetilde{H}_t
$$
• 重置⻔有助于捕捉时间序列⾥短期的依赖关系；  
• 更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。    
### 载入数据集


```python
import os
os.listdir('/home/kesci/input')
```




    ['jaychou_lyrics4703', 'd2l_jay9460']




```python
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


```


```python
import sys
sys.path.append("../input/")
import d2l_jay9460 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
```

### 初始化参数


```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():  
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32) #正态分布
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
     
    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

def init_gru_state(batch_size, num_hiddens, device):   #隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

    will use cpu
    

### GRU模型


```python
def gru(inputs, state, p arams):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

### 训练模型


```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
```


```python
d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
```

    epoch 40, perplexity 150.546513, time 1.06 sec
     - 分开 我想你你的爱爱人 我想你的让我不想想想想想想你想你想想想想想想你想你想想想想想想你想你想想想想想想
     - 不分开 我想你你的爱爱人 我想你的让我想想想想想想你想你想想想想想想你想你想想想想想想你想你想想想想想想你
    epoch 80, perplexity 33.542871, time 1.03 sec
     - 分开 我想要你的微笑 像果我 别你的美笑 你说在我 你是我 别你的生笑 让我想这样 我不要 我不了再想 
     - 不分开 我不能再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我
    epoch 120, perplexity 5.036014, time 1.08 sec
     - 分开 我想要这样牵着你的手不放开 爱可不可以简简单单没有伤害 我 想要你的脑袋在西元前 深埋在美索不达米
     - 不分开  没有你在我有多烦 难道你手不会痛吗 不要再这样打我妈妈 难道你手不会痛吗 我想要你 这样的甜面 
    epoch 160, perplexity 1.466394, time 1.07 sec
     - 分开 我后悔 让不知再想 我不 我不要再想你 爱情来的太快就像龙卷风 不能承受我已无处可躲 我不要再想 
     - 不分开  后后回面我听多烦 我说你没你打我妈妈 爱道你手 我一定好吗活 我该不觉 我跟了这节奏 后知后觉 
    

### 简洁实现


```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

    epoch 40, perplexity 1.016049, time 0.77 sec
     - 分开 我该好好生活 我该好好生活 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋
     - 不分开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专
    epoch 80, perplexity 1.010651, time 0.80 sec
     - 分开 我真的看不下去 以为我较细汉 从小到大只有妈妈的温暖  为什么我爸爸 那么凶 如果真的我有一双翅膀
     - 不分开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专
    epoch 120, perplexity 1.009458, time 0.73 sec
     - 分开 穿梭时间的画面的钟 从反方向开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻 我的伤
     - 不分开始打我妈妈 难道你手不会痛吗 其实我回家就想要阻止一切 让家庭回到过去甜甜 温馨的欢乐香味 虽然这是
    epoch 160, perplexity 1.010228, time 0.81 sec
     - 分开 黑色幽默 说散 你想很久了吧? 我的认真败给黑色幽默 走过了很多地方 我来到伊斯坦堡 就像是童话故
     - 不分开始 担心今天的你过得好不好 整个画面是你 想你想的睡不著 嘴嘟嘟那可爱的模样 还有在你身上香香的味道
    

# LSTM

** 长短期记忆long short-term memory **:  
遗忘门:控制上一时间步的记忆细胞 
输入门:控制当前时间步的输入  
输出门:控制从记忆细胞到隐藏状态  
记忆细胞：⼀种特殊的隐藏状态的信息的流动  


![Image Name](https://cdn.kesci.com/upload/image/q5jk2bnnej.png?imageView2/0/w/640/h/640)

$$
I_t = σ(X_tW_{xi} + H_{t−1}W_{hi} + b_i) \\
F_t = σ(X_tW_{xf} + H_{t−1}W_{hf} + b_f)\\
O_t = σ(X_tW_{xo} + H_{t−1}W_{ho} + b_o)\\
\widetilde{C}_t = tanh(X_tW_{xc} + H_{t−1}W_{hc} + b_c)\\
C_t = F_t ⊙C_{t−1} + I_t ⊙\widetilde{C}_t\\
H_t = O_t⊙tanh(C_t)
$$


### 初始化参数


```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
    
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
    
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), 
            torch.zeros((batch_size, num_hiddens), device=device))
```

    will use cpu
    

### LSTM模型


```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
```

### 训练模型


```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
```

    epoch 40, perplexity 210.587440, time 1.26 sec
     - 分开 我不的我 我不的我 我不不的 我不的我 我不不的 我不的我 我不不的 我不的我 我不不的 我不的我
     - 不分开 我不不的我 我不不的我 我不不的我 我不不的我 我不不的我 我不不的我 我不不的我 我不不的我 我
    epoch 80, perplexity 66.451111, time 1.23 sec
     - 分开 我想你你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想
     - 不分开 我想要你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想想你你的爱我 想
    epoch 120, perplexity 15.650249, time 1.23 sec
     - 分开 我想你你的微笑 一发抖 你给我 说你  说怎怎么对着我 甩开我 别怪我 别你怎么 我该就这生 我不
     - 不分开 我不要 你你的睛我 有发去 快给我 说你怎么 我有就这生 我不好好生活 不知不觉 我该了这生活 我
    epoch 160, perplexity 3.946623, time 1.27 sec
     - 分开 你是我 是是是一脚江 干真看斤的牛肉 我说店小二 三两银够不够 景色入秋 漫天黄沙凉过 塞北的客栈
     - 不分开 我已让我不起 不隔歌人 是谁在在停留 哼哼哈兮 快使用双截棍 哼哼哈兮 快使我有轻功 飞檐走壁 快
    

### 简洁实现


```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

    epoch 40, perplexity 1.019487, time 0.84 sec
     - 分开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻 我的伤口被你拆封 誓言太沉重泪被纵容 
     - 不分开 我来到 你叫我怎么跟你像 不要再这样打我妈妈 我说的话 你甘会听 不要再这样打我妈妈 难道你手不会
    epoch 80, perplexity 1.013503, time 0.86 sec
     - 分开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻 我的伤口被你拆封 誓言太沉重泪被纵容 
     - 不分开 我来到伊斯坦堡 就像是童话故事  有教堂有城堡 每天忙碌地的寻找 到底什么我想要 却发现迷了路怎么
    epoch 120, perplexity 1.010049, time 0.85 sec
     - 分开的玩笑 想通 却又再考倒我 说散 你想很久了吧? 败给你的黑色幽默 说散 你想很久了吧? 我的认真败
     - 不分开 我来到伊斯坦堡 就像是童话故事  有教堂有城堡 每天忙碌地的寻找 到底什么我想要 却发现迷了路怎么
    epoch 160, perplexity 1.011301, time 0.82 sec
     - 分开的督二脉 干什么 干什么 东亚病夫的招牌 干什么 干什么 已被我一脚踢开 快使用双截棍 哼哼哈兮 快
     - 不分开 我不能 爱情走的太快就像龙卷风 不能承受我已无处可躲 我不要再想 我不要再想 我不 我不 我不要再
    

# 深度循环神经网络  

![Image Name](https://cdn.kesci.com/upload/image/q5jk3z1hvz.png?imageView2/0/w/320/h/320)


$$
\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)} + \boldsymbol{b}_h^{(1)})\\
\boldsymbol{H}_t^{(\ell)} = \phi(\boldsymbol{H}_t^{(\ell-1)} \boldsymbol{W}_{xh}^{(\ell)} + \boldsymbol{H}_{t-1}^{(\ell)} \boldsymbol{W}_{hh}^{(\ell)} + \boldsymbol{b}_h^{(\ell)})\\
\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hq} + \boldsymbol{b}_q
$$


```python


num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

    epoch 40, perplexity 40.617305, time 1.21 sec
     - 分开我的让我疯狂 我爱女人 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女
     - 不分开不女人 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我爱女我 我
    epoch 120, perplexity 1.024744, time 1.26 sec
     - 分开爱你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你
     - 不分开已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活 不
    epoch 160, perplexity 1.013805, time 1.32 sec
     - 分开爱你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你
     - 不分开已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活 不
    


```python
gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=6)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

    epoch 40, perplexity 277.322266, time 17.80 sec
     - 分开                                                  
     - 不分开                                                  
    epoch 80, perplexity 276.855183, time 18.41 sec
     - 分开                                                  
     - 不分开                                                  
    

# 双向循环神经网络 

![Image Name](https://cdn.kesci.com/upload/image/q5j8hmgyrz.png?imageView2/0/w/320/h/320)

$$ 
\begin{aligned} \overrightarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(f)} + \overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{hh}^{(f)} + \boldsymbol{b}_h^{(f)})\\
\overleftarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(b)} + \overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{hh}^{(b)} + \boldsymbol{b}_h^{(b)}) \end{aligned} $$
$$
\boldsymbol{H}_t=(\overrightarrow{\boldsymbol{H}}_{t}, \overleftarrow{\boldsymbol{H}}_t)
$$
$$
\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q
$$


```python
num_hiddens=128
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```


```python

```
