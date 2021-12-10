#!/usr/bin/env bash

TWITTER_DATA_PATH=data/twitter_emotion
mkdir -p $TWITTER_DATA_PATH

download_twitter_data() {
    if [ -f $TWITTER_DATA_PATH/$1-ratings-0to1.train.txt ]; then
        echo "Twitter emotion intensity dataset $1 downloaded"
    else
        echo "Twitter emotion intensity dataset $1 is not found, downloading..."
        cd $TWITTER_DATA_PATH
        # train dataset
        wget http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/$1-ratings-0to1.train.txt
        # valid dataset (with and without intensity label)
        wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data/$1-ratings-0to1.dev.target.txt
        wget http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/$1-ratings-0to1.dev.gold.txt
        # test dataset (with and without intensity label)
        wget http://saifmohammad.com/WebDocs/EmoInt%20Test%20Data/$1-ratings-0to1.test.target.txt
        wget http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/$1-ratings-0to1.test.gold.txt
        cd ../..
    fi
}
# download twitter emotion dataset in with all emotions
for emo in anger sadness joy fear; do
  download_twitter_data $emo
done


ISEAR_PATH=data/py_isear_dataset
# https://github.com/sinmaniphel/py_isear_dataset
# 1-joy, 2-fear, 3-anger, 4-sadness, 5-disgust, 6-shame, 7-guilt
if [ -d $ISEAR_PATH ]; then
    echo "ISEAR data downloaded"
else
    echo "ISEAR data not downloaded, downloading right now ... "
    cd data
    git clone https://github.com/sinmaniphel/py_isear_dataset.git
    cd ..
fi

#SEMEVAL_PATH=data/semeval2017
#mkdir -p $SEMEVAL_PATH
#if [ -f $SEMEVAL_PATH/isear.csv ]; then
#    echo "semeval2017 data downloaded"
#else
#    echo "semeval2017 data not downloaded, downloading right now ... "
#    cd $SEMEVAL_PATH
#    wget https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip
#    unzip 2017_English_final.zip
#    rm -rf __MACOSX
#
#    cd ../../
#fi

#EMOBANK_PATH=data/EmoBank
## https://github.com/JULIELab/EmoBank.git
#if [ ! -d $EMOBANK_PATH ]; then
#  echo "Cloning EmoBank Repo ... "
#  cd data
#  git clone https://github.com/JULIELab/EmoBank.git
#  cd ../
#fi

AFFECT_DATA_PATH=data/affect_data
# http://people.rc.rit.edu/%7Ecoagla/affectdata/index.html
if [ ! -d $AFFECT_DATA_PATH ]; then
  echo "downloading affect data high agreement files ... "
  mkdir -p $AFFECT_DATA_PATH
  cd $AFFECT_DATA_PATH
  wget http://people.rc.rit.edu/%7Ecoagla/affectdata/Potter167All4labsagree.txt
  wget http://people.rc.rit.edu/%7Ecoagla/affectdata/hcand460All4labsagree.txt
  wget http://people.rc.rit.edu/%7Ecoagla/affectdata/GrimmsAll4labsagree.txt
  cd ../../
fi

DD_PATH=data/daily_dialog
# https://aclanthology.org/I17-1099/
if [ ! -d $DD_PATH ]; then
  echo "downloading daily dialog dataset ... "
  mkdir -p $DD_PATH
  cd $DD_PATH
  wget https://aclanthology.org/attachments/I17-1099.Datasets.zip
  unzip I17-1099.Datasets.zip
  rm rm -rf __MACOSX I17-1099.Datasets.zip
  mv EMNLP_dataset/* .
  rm -r EMNLP_dataset
  cd ../../
fi

CF_PATH=data/crowdflower
# https://data.world/crowdflower/sentiment-analysis-in-text
if [ ! -d CF_PATH ]; then
  echo "Ask Peter for the dataset ... "
fi

# grounded emotion 2557 tweets with #sad or #happy tag, data does not contain tweet content, need to use twitter API to
# query, might as well just query for even more
#wget http://web.eecs.umich.edu/~mihalcea/downloads/GroundedEmotions.tar.gz


ES_PATH=data/emotion_stimulus
if [ ! -d $ES_PATH ]; then
  echo "downloading daily dialog dataset ... "
  mkdir -p $ES_PATH
  cd $ES_PATH
  wget http://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/Dataset.zip
  unzip Dataset.zip
  rm -r Dataset.zip
  cd ../../
fi


MELD_PATH=data/meld
# https://github.com/declare-lab/MELD
if [ ! -d $MELD_PATH ]; then
  echo "downloading daily dialog dataset ... "
  mkdir -p $MELD_PATH
  cd $MELD_PATH
  wget https://github.com/declare-lab/MELD/raw/master/data/MELD/train_sent_emo.csv
  wget https://github.com/declare-lab/MELD/raw/master/data/MELD/dev_sent_emo.csv
  wget https://github.com/declare-lab/MELD/raw/master/data/MELD/test_sent_emo.csv
  cd ../../
fi

#EmotionPush Registered, hearing back soon

SMILE_PATH=data/smile
if [ ! -d $SMILE_PATH ]; then
  echo "downloading daily dialog dataset ... "
  mkdir -p $SMILE_PATH
  cd $SMILE_PATH
  wget https://figshare.com/ndownloader/articles/3187909/versions/2
  mv "2" Dataset.zip
  unzip Dataset.zip
  rm -r Dataset.zip
  cd ../../
fi