python hydrate_tweet.py -i ../data/ugc/ugc_tweets_annotated.json 
python hydrate_tweet.py -i ../data/schedule/schedule_tweets_annotated.json 

python extract_features.py \
    -i ../data/ugc/ugc_tweets_annotated_summary.csv \
    -e ../data/ugc/ugc_tweets_annotated_entities.csv \
    -o ../data/ugc/weka/ugc_tweets_weka.arff  \
    -n ../data/ugc/emnlp_nn/ugc_tweets_nn.txt

python extract_features.py \
    -i ../data/schedule/schedule_tweets_annotated_summary.csv \
    -e ../data/schedule/schedule_tweets_annotated_entities.csv \
    -o ../data/schedule/weka/schedule_tweets_weka.arff  \
    -n ../data/schedule/emnlp_nn/schedule_tweets_nn.txt


python schedule_matcher.py -w 0.5 -c 0.5 -t 1000 \
    -i ../data/ugc/ugc_tweets_annotated_summary.csv \
    -s ../data/schedule/schedule_tweets_annotated_summary.csv


./conlleval < ../results/schedule_matcher_0.5_0.5_1000.txt \
    > ../results/score.schedule_matcher_0.5_0.5_1000.txt