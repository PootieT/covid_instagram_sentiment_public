from torch.utils.data import Dataset

class sentenceClsData(Dataset):
    def __init__(self, tkdata):
        super(sentenceClsData, self).__init__()
        self._tkdata = tkdata
        self._len = len(self._tkdata['labels'])
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        return {
                'input_ids': self._tkdata['input_ids'][i],
#                 'token_type_ids': self._tkdata['token_type_ids'][i],
                'attention_mask': self._tkdata['attention_mask'][i],
                'labels': self._tkdata['labels'][i]
               }