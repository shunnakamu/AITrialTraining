from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa

from tqdm.notebook import tqdm

from transformers import Wav2Vec2Processor, Wav2Vec2Model


# https://huggingface.co/transformers/model_doc/wav2vec2.html#wav2vec2forctc
# processorとmodelを使う場合、誤差逆伝播しようと思うと実装が難しそうなので、上記から修正したほうが良さそう

class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_size=512, num_classes=8, device='cuda', sr=16000):
        super(Wav2VecClassifier, self).__init__()
        self.sr = sr
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.lstm = nn.LSTM(768, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        input_values = self.processor(x, return_tensors="pt", sampling_rate=self.sr).input_values
        # input_values = torch.squeeze(input_values)
        input_values = torch.squeeze(input_values.permute(2, 1, 0)).to(self.device)
        hidden_states = self.model(input_values).last_hidden_state
        out, _ = self.lstm(hidden_states)
        out = F.relu(out)
        out = self.fc(out)
        return F.softmax(out)


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
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=1):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## get sequence lengths
        lengths = torch.tensor([len(review_w2v) for review_w2v, star in batch])
        ## padd
        batch = [torch.Tensor(review_w2v) for review_w2v, star in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch)
        return batch, lengths

    def __call__(self, batch):
        return self.pad_collate(batch)


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
