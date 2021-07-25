from collections import defaultdict
from numpy.lib.function_base import _parse_input_dimensions
from tokenizers import ByteLevelBPETokenizer
from model import Model 
from sklearn.model_selection import StratifiedKFold 
from preprocess_data import preprocessing
from transformers import AdamW,get_linear_schedule_with_warmup
from data import TweetSentimentExtraction, MyCollate
from torch.utils.data import DataLoader
import torch 
from utils import train_model
import gc 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--max_len',type=int,default=128,help='max len sequence')
parser.add_argument('--file_train',type=str,default="Data/train.csv")
parser.add_argument('--file_test',type=str,default="Data/test.csv")
parser.add_argument('--file_sub',type=str,default="Data/sample_submission.csv")
parser.add_argument('--num_warmup_steps',type=int,default=50)   
parser.add_argument('--batch_size',type=int,required=True)
parser.add_argument('--pretrain',type=str,required=True,help='Name pretrained model')
parser.add_argument('--lr',type=float,required=True,help='learning rate for optimizer')
parser.add_argument('--n_fold',type=int,required=True,help='number fold when training')
parser.add_argument('--n_epochs',type=int,required=True)

arg=parser.parse_args()
gc.enable()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Config:
    MAX_LENGTH=arg.max_len
    BATCH_SIZE=arg.batch_size
    EPOCHS=arg.n_epochs
    TOKENIZER=ByteLevelBPETokenizer(
      vocab=f'{arg.pretrain}/vocab.json',
      merges=f'{arg.pretrain}/merges.txt',
      add_prefix_space=True,
      lowercase=True
    )
    PATH_FILE_TRAIN=arg.file_train
    PATH_FILE_TEST=arg.file_test
    PATH_FILE_SUB=arg.file_sub
    TOKENIZER.enable_truncation(max_length=MAX_LENGTH)
    VOCAB=TOKENIZER.get_vocab()
    INT_TO_WORD={value:key for key,value in VOCAB.items()}
    CLS_ID=VOCAB['<s>']
    SEP_ID=VOCAB['</s>']
    PAD_ID=VOCAB['<pad>']


def get_train_val_loader(df,train_index,val_index):
    train=df.iloc[train_index]
    val=df.iloc[val_index]
    train_dataset=TweetSentimentExtraction(train,Config)
    val_dataset=TweetSentimentExtraction(val,Config)
    train_loader=DataLoader(train_dataset,batch_size=Config.BATCH_SIZE,shuffle=False,num_workers=2,collate_fn=MyCollate(Config.PAD_ID))
    val_loader=DataLoader(val_dataset,batch_size=Config.BATCH_SIZE,shuffle=False,num_workers=2,collate_fn=MyCollate(Config.PAD_ID))
    return train_loader,val_loader

train=preprocessing(Config)
kfold=StratifiedKFold(n_splits=arg.n_fold,shuffle=True,random_state=43)

for fold,(train_indexs,val_indexs) in enumerate(kfold.split(train,train.sentiment),start=1):
    model=Model(arg.pretrain).to(device)
    optimizer=AdamW(model.parameters(),lr=arg.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=arg.num_warmup_steps,
        num_training_steps=len(train)//Config.BATCH_SIZE*Config.EPOCHS    
    )

    train_loader,val_loader=get_train_val_loader(train,train_indexs,val_indexs)
    train_model(model,train_loader,val_loader,optimizer,scheduler,fold,len(val_indexs),Config)
    del model,optimizer,scheduler
    torch.cuda.empty_cache()
    gc.collect()
  