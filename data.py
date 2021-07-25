import torch 
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from preprocess_data import get_data

class TweetSentimentExtraction(Dataset):
    def __init__(self,data,config):
        self.data=data
        self.tokenizer=config.TOKENIZER
        self.max_length=config.MAX_LENGTH
        self.is_test="selected_text" in self.data
        self.config=config
        self.cls_id=self.config.VOCAB['<s>']
        self.sep_id=self.config.VOCAB['</s>']
        self.pad_id=self.config.VOCAB['<pad>']


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        row=self.data.iloc[idx]
        tweet=row.text
        sentiment=row.sentiment
        tweet,input_ids,attention_mask,offsets=get_data(tweet,sentiment,self.config)
        data={}
        data['input_ids']=torch.tensor(input_ids,dtype=torch.long)
        data['attention_mask']=torch.tensor(attention_mask,dtype=torch.long)
        data['tweet']=tweet
        data['offsets']=offsets
        if self.is_test:
            start_index=row.start_index
            end_index=row.end_index
            data['start_index']=start_index
            data['end_index']=end_index
        return data
    


class MyCollate:
    def __init__(self,pad_id,is_test=False):
        self.pad_id=pad_id
        self.is_test=is_test
        
    def __call__(self,batch):
        input_ids=[item['input_ids'] for item in batch]
        attention_mask=[item['attention_mask'] for item in batch]
        tweet=[item['tweet'] for item in batch]
        offsets=[item['offsets'] for item in batch]
        input_ids=pad_sequence(input_ids,batch_first=True,padding_value=self.pad_id)
        attention_mask=pad_sequence(attention_mask,batch_first=True,padding_value=0)
        if len(offsets)<input_ids.size(1):
          padding_len=input_ids.size(1)-len(offsets)
          offsets=offsets+[(0,0)]*padding_len

        if not self.is_test:
            start_index=[item['start_index'] for item in batch]
            end_index=[item['end_index'] for item in batch]
            start_index=torch.tensor(start_index,dtype=torch.long)
            end_index=torch.tensor(end_index,dtype=torch.long)
            return {
                "input_ids":input_ids,
                "attention_mask":attention_mask,
                "offsets":offsets,
                'tweet':tweet,
                "start_index":start_index,
                "end_index":end_index
            }
        else:
            return {
                "input_ids":input_ids,
                "attention_mask":attention_mask,
                "offsets":offsets,
                'tweet':tweet
            }
        