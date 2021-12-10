from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizer, DataCollatorWithPadding

from dataset import sentenceClsData
from recipes import Roberta
from utils import readData
import torch

train_pth = '/projectnb/cs640g/students/pranchan/covid_instagram_sentiment/data/twitter_emotion/anger-ratings-0to1.train.txt'
val_pth = '/projectnb/cs640g/students/pranchan/covid_instagram_sentiment/data/twitter_emotion/anger-ratings-0to1.dev.gold.txt'

train = readData(train_pth)
val = readData(val_pth)


X_train, y_train = zip(*train)
X_val, y_val = zip(*val)

checkpoint = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)

token_train = tokenizer(X_train, truncation=True)
token_test = tokenizer(X_val, truncation=True)

token_train['labels'] = y_train
token_test['labels'] = y_val

train_data = sentenceClsData(token_train)
test_data  = sentenceClsData(token_test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    train_data, shuffle=True, batch_size=4, collate_fn=data_collator, num_workers=4
)
eval_dataloader = DataLoader(
    test_data, batch_size=16, collate_fn=data_collator, num_workers=4
)


model = Roberta(checkpoint, 1, train_dataloader, eval_dataloader)
trainer = pl.Trainer(gpus=1, max_epochs=3, auto_scale_batch_size='power')
trainer.fit(model)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()