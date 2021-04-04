"""
改良版がこちらにあるのでこちらを参照。
https://github.com/tsubauaaa/AITrialTraining/blob/main/Training10/AITraining10-2.ipynb
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa
import numpy as np
import keras
import math


from tqdm.notebook import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_size=512, num_classes=8, device='cpu', sr=16000):
        super(Wav2VecClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.sr = sr
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.lstm = nn.LSTM(768, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        input_values = self.processor(x, return_tensors="pt", sampling_rate=self.sr).input_values
        # (batch_size, seq_len)に次元入れ替え(batch first対応)
        input_values = torch.squeeze(input_values.permute(1, 2, 0)).to(self.device)
        hidden_states = self.model(input_values).last_hidden_state

        # pooling
        # LSTMではなくシーケンス方向に平均にしてしまう (予定)

        # lstm_out, lstm_hidden = self.lstm(hidden_states)
        # lstm_hiddenの最後をhidden_sizeに平す
        # out = self.fc(lstm_hidden[0].view(-1, self.hidden_size))
        out = F.relu(out)
        out = F.softmax(out)
        return out


class AudioRawDataset(torch.utils.data.Dataset):
    def __init__(self, audio_file_list, transform=None, num_classes=8, sr=16000):
        self.transform = transform
        self.file_path_list = audio_file_list
        self.label = [int(x.name.split('-')[2]) - 1 for x in audio_file_list]
        self.sr = sr

        self.datanum = len(self.label)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        audio_file_path = self.file_path_list[idx]
        out_label = self.label[idx]
        out_data, fs = librosa.load(audio_file_path, sr=self.sr)

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label


class PadCollate:
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''
    def __init__(self,test=False,percentile=100):
        self.test = test
        self.percentile = percentile
    def __call__(self, batch):
        if not self.test:
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
        else:
            data = batch
        lens = [len(x) for x in data]
        max_len = np.percentile(lens, self.percentile)
        data = keras.preprocessing.sequence.pad_sequences(data, maxlen=int(max_len))
        data = torch.tensor(data, dtype=torch.float32)
        if not self.test:
            target = torch.tensor(target,dtype=torch.int64)
            return [data,target]
        return [data]


if __name__ == '__main__':
    audio_dir = Path('RAVDESS')
    audio_files = audio_dir.glob("**/*.wav")
    audio_file_list = list(audio_files)

    audio_dataset = AudioRawDataset(audio_file_list)
    n_samples = len(audio_dataset)
    train_size = int(len(audio_dataset) * 0.8)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(audio_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=PadCollate())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=PadCollate())

    device = torch.device('cuda')
    model = Wav2VecClassifier(device=device)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    n_epochs = 10

    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        total_size = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), total_loss / total_size))
