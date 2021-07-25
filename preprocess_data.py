import pandas as pd 

def read_data(path_file):
    data=pd.read_csv(path_file)
    return data 

def get_data(tweet,sentiment,config):
    tweet=" "+" ".join(tweet.lower().split())
    token=config.TOKENIZER.encode(tweet)
    sentiment_values={
      value:config.VOCAB[value] for value in ['positive','negative','neutral']
    }
    input_ids=[config.CLS_ID]+[sentiment_values[sentiment]]+[config.SEP_ID]+token.ids+[config.SEP_ID]
    attention_mask=[1]*len(input_ids)
    offsets=[(0,0)]*3+token.offsets+[(0,0)]
    return tweet,input_ids,attention_mask,offsets


def find_index(tweet,selected_text,offsets):
    selected_text=" "+" ".join(selected_text.lower().split())
    index1,index2=None,None
    length=len(selected_text)-1
    for value in [position for position,value in enumerate(tweet) if value==selected_text[1]]:
        if " "+tweet[value:value+length]==selected_text:
            index1=value
            index2=value+length-1
    temp=[0]*len(tweet)
    start_index,end_index=None,None
    if index1!=None and index2!=None:
        for i in range(index1,index2+1):
            temp[i]=1
        list_index=[]
        for i,(offset1,offset2) in enumerate(offsets):
            if sum(temp[offset1:offset2])>0:
                list_index.append(i)
        start_index=list_index[0]
        end_index=list_index[-1]
    return start_index,end_index


def preprocessing(config):
    data=read_data(config.PATH_FILE_TRAIN)
    data=data.dropna()
    data=data.reset_index(drop=True)
    start_indexs,end_indexs=[],[]
    for i in range(len(data)):
        tweet_temp=data['text'][i]
        sentiment=data['sentiment'][i]
        selected_temp=data['selected_text'][i]
        tweet_temp,_,_,offsets=get_data(tweet_temp,sentiment,config)
        start_index,end_index=find_index(tweet_temp,selected_temp,offsets)
        if start_index is None:
            raise Exception('None index')
            #print(tweet_temp,'------>',selected_temp)
        start_indexs.append(start_index)
        end_indexs.append(end_index)

    data['start_index']=start_indexs
    data['end_index']=end_indexs
    return data 
