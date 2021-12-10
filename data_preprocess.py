import json
import os
import random
from typing import Dict, Optional, List

import fasttext
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from data.py_isear_dataset.py_isear.isear_loader import IsearLoader
from enums import EMOT
from utils import remove_emoji, translate_text


def filter_isear_data():
    attributes = ['EMOT', 'SIT']
    target = ['TROPHO', 'TEMPER']
    loader = IsearLoader(attributes, target, True)
    data = loader.load_isear('data/py_isear_dataset/isear.csv')
    df = pd.DataFrame(data.get_data(), columns=["emotion"])
    df["emotion"] = df["emotion"].map({idx: emotion.lower() for idx, emotion in enumerate(EMOT)})
    df["text"] = data.get_freetext_content()
    df = df[df["emotion"].isin(["anger", "joy", "sadness", "fear"])]
    df.to_csv("data/py_isear_dataset/isear_filtered.csv", index=False, header=None)


def filter_affect_data():
    story_files = ["GrimmsAll4labsagree.txt", "hcand460All4labsagree.txt", "Potter167All4labsagree.txt"]
    df = pd.DataFrame(columns=["emotion", "text"])
    emo_dict = {2: "anger", 3: "fear", 4: "joy", 6: "sadness"}
    for file_path in story_files:
        with open(f"data/affect_data/{file_path}") as f:
            for line in f:
                if not line.endswith(".agree\n"):
                    data = line.strip().split("@")
                    if int(data[1]) in emo_dict:
                        df = df.append({"emotion": emo_dict[int(data[1])], "text": data[2]}, ignore_index=True)

    df.to_csv("data/affect_data/affect_filtered.csv", index=False, header=None)


def filter_daily_dialog_data():
    # merge disgust with anger
    emo_dict = {1: "anger", 2: "anger", 3: "fear", 4: "joy", 5: "sadness"}
    df = pd.DataFrame(columns=["emotion", "text"])
    with open("data/daily_dialog/dialogues_text.txt", "r") as text_f:
        with open("data/daily_dialog/dialogues_emotion.txt", "r") as emo_f:
            for text_line in tqdm(text_f):
                emo_line = next(emo_f)
                emo_list = emo_line.strip().split(" ")
                text_list = text_line.strip().split(" __eou__ ")
                for i, emo in enumerate(emo_list):
                    if int(emo) in emo_dict:
                        df = df.append({"emotion": emo_dict[int(emo)], "text": text_list[i].strip("__eou__")}, ignore_index=True)

    df.to_csv("data/daily_dialog/dialog_filtered.csv", index=False, header=None)


def filter_crowdflower():
    emo_dict = {'empty': "sadness", 'sadness':"sadness", 'enthusiasm':"joy", 'worry': "fear", 'love':"joy", 'fun': "joy",
     'hate': "anger", 'happiness': "joy", 'relief': "joy", 'anger': "anger"}  # remove 'neutral', 'surprise', 'boredom'
    df = pd.read_csv("data/crowdflower/text_emotion.csv")
    df = df[df["sentiment"].isin(emo_dict.keys())]
    df["sentiment"] = df["sentiment"].map(emo_dict)
    df = df.drop(columns=["tweet_id", "author"])
    df.to_csv("data/crowdflower/text_emotion_filtered.csv", index=False, header=None)


def filter_emotion_stimulus():
    df = pd.DataFrame(columns=["emotions", "text"])
    emo_dict = {"happy": "joy", "sad": "sadness", "disgust":"anger", "anger":"anger", "fear": "fear", "shame": "sadness"}  # omit surprise
    with open("data/emotion_stimulus/Dataset/Emotion Cause.txt", "r") as f:
        for line in f:
            emotion = line[1:line.find(">")]
            if emotion in emo_dict:
                line = line.replace(f"<{emotion}>", "").replace(f"<\\{emotion}>", "")
                line = line.replace("<cause>", "").replace("<\cause>", "")
                df = df.append({"emotions": emo_dict[emotion], "text": line.strip()}, ignore_index=True)

    with open("data/emotion_stimulus/Dataset/No Cause.txt", "r") as f:
        for line in f:
            emotion = line[1:line.find(">")]
            if emotion in emo_dict:
                line = line.replace(f"<{emotion}>", "").replace(f"<\\{emotion}>", "")
                df = df.append({"emotions": emo_dict[emotion], "text": line.strip()}, ignore_index=True)

    df.to_csv("data/emotion_stimulus/emotion_cause_filtered.csv", index=False, header=None)


def filter_meld_dataset():
    emo_dict = {"joy": "joy", "sadness": "sadness", "disgust": "anger", "anger": "anger", "fear": "fear"}  # omit surprise and neutral

    df = pd.read_csv("data/meld/dev_sent_emo.csv")
    df = df.append(pd.read_csv("data/meld/train_sent_emo.csv"))
    df = df.append(pd.read_csv("data/meld/test_sent_emo.csv"))
    df = df.drop(columns=["Sr No.", "Speaker", "Sentiment", "Sentiment","Dialogue_ID","Utterance_ID","Season","Episode","StartTime","EndTime"])
    df = df[["Emotion", "Utterance"]]
    df = df[df["Emotion"].isin(emo_dict.keys())]
    df["Emotion"] = df["Emotion"].map(emo_dict)
    df.to_csv("data/meld/meld_filtered.csv", index=False, header=None)


def filter_smile_dataset():
    emo_dict = {"happy": "joy", "happy|surprise": "joy", "happy|sad": "joy",
                "sad|disgust": "sadness", "sad|disgust|angry": "anger", "sadness": "sadness",
                "sad|angry": "anger", "disgust|angry": "anger", "disgust": "anger", "angry": "anger",
                }  # omit surprise, not-relevant, and nocode

    df = pd.read_csv("data/smile/smile-annotations-final.csv", names=["id", "text", "emotions"])
    df = df.drop(columns=["id"])
    df = df[["emotions", "text"]]
    df = df[df["emotions"].isin(emo_dict.keys())]
    df["emotions"] = df["emotions"].map(emo_dict)
    df.to_csv("data/smile/smile_filtered.csv", index=False, header=None)


def filter_emotion_push_dataset():
    with open("data/emotion_push/EmotionPush/emotionpush.json") as f:
        data_train = json.load(f)
    with open("data/emotion_push/emotionpush_eval_gold.json") as f:
        data_eval = json.load(f)

    emo_dict = {"joy": "joy", "sadness": "sadness", "disgust": "anger", "anger": "anger",
                "fear": "fear"}  # omit surprise, neutral, and non-neutral

    df = pd.DataFrame(columns=["emotions", "text"])
    for data in [data_train, data_eval]:
        for conv in data:
            for sent in conv:
                if sent["emotion"] in emo_dict:
                    df = df.append({"emotions": emo_dict[sent["emotion"]], "text": sent["utterance"]}, ignore_index=True)

    df.to_csv("data/emotion_push/emotion_push_filtered.csv", index=False, header=None)

def filter_twitter_emotion_dataset():
    """
    convert twitter emotion intensity dataset to classification by using tweets with intensity >= 0.5
    :return:
    """
    data_dir = "data/twitter_emotion"
    for split in ["train", "dev", "test"]:
        df = pd.DataFrame(columns=["id", "text", "emotions", "intensity"])
        for emotion in ["anger", "fear", "joy", "sadness"]:
            file = f"{emotion}-ratings-0to1.{split}{'.gold' if split != 'train' else ''}.txt"
            df = df.append(pd.read_csv(f"{data_dir}/{file}", sep="\t", names=["id", "text", "emotions", "intensity"]))
        df = df[df["intensity"] >= 0.5]
        df = df.drop(columns=["id", "intensity"])
        df = df[["emotions", "text"]]
        df.to_csv(f"{data_dir}/{split}_filtered.csv", index=False, header=None)


def combine_datasets(
    isear=True, affect=True, dd=True, crowdflower=True, stimulus=True, meld=True, smile=True, push=True
):
    twit_dir = "data/twitter_emotion"
    for split in ["dev", "test"]:
        src = f"{twit_dir}/{split}_filtered.csv"
        dst = f"data/{split}_filtered.csv"
        copyfile(src, dst)

    df = pd.read_csv(f"{twit_dir}/train_filtered.csv", names=["emotions", "text"])
    if isear:
        df = df.append(pd.read_csv(f"data/py_isear_dataset/isear_filtered.csv", names=["emotions", "text"]))
    if affect:
        df = df.append(pd.read_csv(f"data/affect_data/affect_filtered.csv", names=["emotions", "text"]))
    if dd:
        df = df.append(pd.read_csv(f"data/daily_dialog/dialog_filtered.csv", names=["emotions", "text"]))
    if crowdflower:
        df = df.append(pd.read_csv(f"data/crowdflower/text_emotion_filtered.csv", names=["emotions", "text"]))
    if stimulus:
        df = df.append(pd.read_csv(f"data/emotion_stimulus/emotion_cause_filtered.csv", names=["emotions", "text"]))
    if meld:
        df = df.append(pd.read_csv(f"data/meld/meld_filtered.csv", names=["emotions", "text"]))
    if smile:
        df = df.append(pd.read_csv(f"data/smile/smile_filtered.csv", names=["emotions", "text"]))
    if push:
        df = df.append(pd.read_csv(f"data/emotion_push/emotion_push_filtered.csv", names=["emotions", "text"]))

    df.to_csv(f"data/train_filtered.csv", index=False, header=None)
    print("data statistics", df.groupby("emotions").count())


def process_instagram_dataset():
    if not os.path.exists("models/lid.176.ftz"):
        os.makedirs("models")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        r = requests.get(url)
        open("models/lid.176.ftz", 'wb').write(r.content)
    df = pd.read_excel("data/instagram/Labeled_instagram_posts_related_to_covid.xlsx")
    model = fasttext.load_model("models/lid.176.ftz")
    df["Contents"] = df["Contents"].map(lambda text: ' '.join(str(text).replace("\n", " ").replace("_x000D_"," ").split()))
    df["fasttext"] = df["Contents"].map(lambda text: model.predict(remove_emoji(text.replace("#"," "))))
    df["language"] = df["fasttext"].map(lambda x: x[0][0].replace("__label__", ""))
    df["language_confidence"] = df["fasttext"].map(lambda x: x[1][0])
    lang_stat = df["language"].value_counts()
    print("language statsitics:", lang_stat)
    language_weights = {k:v for k,v in zip(lang_stat.index, lang_stat)}

    val, weight = zip(*[(k, v) for k, v in language_weights.items()])
    plt.figure()
    plt.bar(val[:20], weight[:20])
    plt.yscale = "log"
    plt.xlabel = "Languages"
    plt.ylabel = "Post Count"
    plt.title = "Instagram Post Langauge Distribution"
    plt.xticks(rotation='vertical')
    # plt.show()
    plt.savefig("instagram_language_distribution.png")
    df[["language", "Contents"]].to_csv(f"data/instagram/text.csv", index=False, header=None)
    return language_weights


def translate_row(lg: str, text: str):
    if lg == "en":
        return text
    else:
        return translate_text("en", text)["translatedText"]


def translate_instagram_dataset():
    assert os.path.exists(f"data/instagram/text.csv"), "Run process_instagram_dataset first to generate text only csv"
    df = pd.read_csv(f"data/instagram/text.csv", names=["language", "text"])
    trans_df = df.copy()
    for index, row in tqdm(df.iterrows()):
        trans_df.loc[index, "translation"] = translate_row(row["language"], row["text"])
    trans_df.to_csv("data/instagram/translated_text.csv", index=False, header=None)


def back_translation_augmentation(df:pd.DataFrame, num_samples: int, language_weights=Optional[Dict[str, float]]) -> pd.DataFrame:
    """
    Given a dataframe of sentences, back translate a sample of such data from different languages where
    the proportion of the BT language is determined by the language_weights parameter
    :param df: pandas datafram containint at least text, optionally emotions column
    :param num_samples: number of samples to sample
    :param language_weights:
    :return:
    """
    if num_samples == len(df):
        sample_index = list(range(len(df)))
    else:
        sample_index = random.sample(range(len(df)), num_samples)
    source_samples = df["text"][sample_index].tolist()
    if "emotions" in df.columns:
        source_label = df["emotions"][sample_index].tolist()
    target_languages = random.choices(list(language_weights.keys()), weights=list(language_weights.values()), k=num_samples)
    target_text = pd.DataFrame()
    for i in tqdm(range(num_samples)):
        if target_languages == "en":
            text = source_samples[i]
        else:
            foreign_text = translate_text(target_languages[i], source_samples[i])["translatedText"]
            if foreign_text is not None:
                text = translate_text("en", foreign_text)["translatedText"]
                if text is None:
                    text = source_samples[i]
            else:
                text = source_samples[i]
        if "emotions" in df.columns:
            target_text = target_text.append({"text": text, "emotions": source_label[i]}, ignore_index=True)
        else:
            target_text = target_text.append({"text": text}, ignore_index=True)
    return target_text


def augment_twitter_dev_test_set():
    language_weights = process_instagram_dataset()
    del language_weights["en"]

    dev_path = "data/twitter_emotion/dev_filtered.csv"
    test_path = "data/twitter_emotion/test_filtered.csv"
    dev_df = pd.read_csv(dev_path, names=["emotions", "text"])
    test_df = pd.read_csv(test_path, names=["emotions", "text"])
    new_dev = back_translation_augmentation(dev_df, len(dev_df), language_weights=language_weights)
    new_dev.to_csv("dev_bt.csv", index=False, header=None)
    new_test = back_translation_augmentation(test_df, len(test_df), language_weights=language_weights)
    new_test.to_csv("test_bt.csv", index=False, header=None)


if __name__ == "__main__":
    np.random.seed(42)
    # filter_isear_data()
    # filter_affect_data()
    # filter_daily_dialog_data()
    # filter_crowdflower()
    # filter_emotion_stimulus()
    # filter_meld_dataset()
    # filter_smile_dataset()
    # filter_emotion_push_dataset()
    # filter_twitter_emotion_dataset()
    # combine_datasets()
    # process_instagram_dataset()
    translate_instagram_dataset()
    # augment_twitter_dev_test_set()