import os

models=[ 'lstm',
        'td_lstm',
        'atae_lstm',
        'ian',
        'memnet',
        'ram',
        'cabasc',
        'tnet_lf',
        'aoa',
        'mgan',
        'bert_spc',
        'aen_bert',
        'lcf_bert',
        'keaen_bert',]

for model in models:
    if 'bert' in model:
        os.system('python train.py --model_name '+model+' --learning_rate '+'2e-5'+'--num_epoch 2')
    else:
        os.system('python train.py --model_name '+model+' --learning_rate '+'1e-3'+'--num_epoch 10')
    print(model+' has down！')