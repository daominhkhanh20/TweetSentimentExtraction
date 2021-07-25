from torch import nn 
import torch 
import os 
from collections import defaultdict
import time 
import numpy as np 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_selected_text(text,start_index,end_index,offsets):
    selected_text=""
    for i in range(start_index,end_index+1):
        selected_text+=text[offsets[i][0]:offsets[i][1]]
        if (i+1)<len(offsets) and offsets[i][0]<offsets[i+1][0]:
            selected_text+=" "
    return selected_text

def get_loss(start_targets,end_targets,start_logit,end_logit):
    loss_fn=nn.CrossEntropyLoss(reduction="mean")
    loss1=loss_fn(start_logit,start_targets)
    loss2=loss_fn(end_logit,end_targets)
    loss=loss1+loss2
    return loss

def jaccard_score(str1,str2):
    a=set(str1.lower().split())
    b=set(str2.lower().split())
    c=a.intersection(b)
    return float(len(c))/(len(a)+len(b)-len(c))

def computer_jaccard_score(text,start_index,end_index,start_logit,end_logit,offsets):
    start_pred=np.argmax(start_logit)
    end_pred=np.argmax(end_logit)
    if start_pred>end_pred:
        pred=text
    else:
        pred=get_selected_text(text,start_pred,end_pred,offsets)
    true=get_selected_text(text,start_index,end_index,offsets)
    return jaccard_score(true,pred)

def save_checkpoint(model_state_dict,fold):
    path="/media/daominhkhanh/D:/Data/Project/TweetSentimentExtraction/Model"
    if os.path.exists(path) is False:
        os.makedirs(path,exist_ok=True)
        
    # with open(path+f'/history_{epoch}_fold{fold}.pickle','wb') as file:
    #     pickle.dump(history,file,protocol=pickle.HIGHEST_PROTOCOL)
    # print("Save history done")
    torch.save(model_state_dict,path+f"/model_fold{fold}.pth")
    print("Save model done")


def evaluate(model,loader,len_val):
    print('---------------------------TIME FOR EVALUATE---------------------------')
    model.eval()
    val_loss=0
    score=0
    with torch.no_grad():
        for idx,data in enumerate(loader):
            input_ids=data['input_ids'].to(device)
            attention_mask=data['attention_mask'].to(device)
            offsets=data['offsets']
            tweets=data['tweet']
            start_logit,end_logit=model(input_ids,attention_mask)
            loss=get_loss(
                data['start_index'].to(device),
                data['end_index'].to(device),
                start_logit,
                end_logit
            )
            val_loss+=loss.item()
            start_indexs=data['start_index'].cpu().detach().numpy()
            end_indexs=data['end_index'].cpu().detach().numpy()
            start_logit=start_logit.cpu().detach().numpy()
            end_logit=end_logit.cpu().detach().numpy()
            for i in range(len(input_ids)):
                score+=computer_jaccard_score(
                    tweets[i],
                    start_indexs[i],
                    end_indexs[i],
                    start_logit[i],
                    end_logit[i],
                    offsets[i]
                )
    return val_loss/len(loader),score/len_val

def train_model(model,train_loader,val_loader,optimizer,scheduler,fold,len_val,config):
    model.train()
    train_loss=0
    history=defaultdict(list)
    jaccard_score_final=None
    print(f"----------------------------------FOLD {fold}----------------------------------\n\n")
    for epoch in range(config.EPOCHS+1):
        train_loss=0
        start_time=time.time()
        print('---------------------------TIME FOR TRANING---------------------------')
        for idx,data in enumerate(train_loader):
            input_ids=data['input_ids'].to(device)
            attention_mask=data['attention_mask'].to(device)
            start_logit,end_logit=model(input_ids,attention_mask)
            optimizer.zero_grad()
            loss=get_loss(
                data['start_index'].to(device),
                data['end_index'].to(device),
                start_logit,
                end_logit
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss+=loss.item()
            if idx%100==0:
                print(idx,end=" ")
        print()
        train_loss/=len(train_loader)
        val_loss,score=evaluate(model,val_loader,len_val)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['jaccard_score'].append(score)
        print(f"Epochs:{epoch}---Train loss:{train_loss}---Val loss:{val_loss}---Jaccard score val:{score}---Time:{time.time()-start_time}")
        if jaccard_score_final is None or score>jaccard_score_final:
          model_state=model.state_dict()
          jaccard_score_final=score
    
    save_checkpoint(model_state,fold)             
    print('\n\n')

def get_start_end_predict(models,input_ids,attention_mask,size_batch):
    start_preds=torch.tensor([0]*size_batch,dtype=torch.long).to(device)
    end_preds=torch.tensor([0]*size_batch,dtype=torch.long).to(device)
    for i in range(len(models)):
        model=models[i]
        start_logit,end_logit=model(input_ids,attention_mask)
        start_index=torch.argmax(start_logit,dim=1)
        end_index=torch.argmax(end_logit,dim=1)
        start_preds+=start_index
        end_preds+=end_index
    start_preds=start_preds.cpu().detach().numpy()
    end_preds=end_preds.cpu().detach().numpy()
    return (start_preds/len(models)).astype(int),(end_preds/len(models)).astype(int)
        

def get_predict(models,test_loader):
    preds=[]
    for data in test_loader:
        input_ids=data['input_ids'].to(device)
        attention_mask=data['attention_mask'].to(device)
        offsets=data['offsets']
        tweets=data['tweet']
        start_indexs,end_indexs=get_predict(models,input_ids,attention_mask,len(tweets))
        for i in range(len(tweets)):
            if start_indexs[i]>end_indexs[i]:
                result=tweets[i]
            else:
                result=get_selected_text(tweets[i],start_indexs[i],end_indexs[i],offsets[i])
            preds.append(result)
    return preds