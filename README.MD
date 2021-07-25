## Tweet Sentiment Extraction in Kaggle Competition([link](https://www.kaggle.com/c/tweet-sentiment-extraction))


## Traning
<pre>
python3 train.py --batch_size 32 [REQUIRED]
                 --pretrain 'roberta-base' [REQUIRED]
                 --lr 3e-5 [REQUIRE]
                 --n_fold 5 [REQUIRED]
                 --n_epochs 10 [REQUIRED]
</pre>

## Result
My Jaccard score is 0.65911 ([link](https://www.kaggle.com/daominhkhanh/tweetsentimentextraction))