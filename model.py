import torch 
from torch import nn 
from transformers import RobertaConfig,RobertaModel

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self,name_pretrain):
        super(Model,self).__init__()
        self.config=RobertaConfig.from_pretrained(f'{name_pretrain}/config.json',output_hidden_states=True,return_dict=True)
        self.bert=RobertaModel.from_pretrained(name_pretrain,config=self.config)
        self.hidden_size=self.bert.config.hidden_size
        self.norm=nn.LayerNorm(self.hidden_size)
        self.linear=nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,2)
        )
    
    def __init__weight(self,module):
        if isinstance(module,nn.Linear):
            module.weight.data.normal_(mean=0,std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        outputs=self.bert(input_ids,attention_mask)
        hidden_states=outputs.hidden_states
        out=torch.stack([hidden_states[-1],hidden_states[-2],hidden_states[-3],hidden_states[-4]])
        out=torch.mean(out,0)
        out=self.linear(out)
        start_logit,end_logit=torch.split(out,1,-1)
        start_logit=start_logit.squeeze(dim=-1)
        end_logit=end_logit.squeeze(dim=-1)
        return start_logit,end_logit