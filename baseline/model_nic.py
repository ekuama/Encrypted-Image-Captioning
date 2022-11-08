import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderA(nn.Module):
    def __init__(self, name):
        super(EncoderA, self).__init__()
        self.name = name
        if self.name == 'ResNet101':
            self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet = torchvision.models.resnet101(pretrained=True)
            modules = list(resnet.children())[1:-1]
            self.resnet = nn.Sequential(*modules)
        elif self.name == 'ResNeXt101':
            self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet = torchvision.models.resnext101_32x8d(pretrained=True)
            modules = list(resnet.children())[1:-1]
            self.resnet = nn.Sequential(*modules)
        else:
            self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet = torchvision.models.resnet50(pretrained=True)
            modules = list(resnet.children())[1:-1]
            self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, phase, amp):
        images = torch.cat([phase, amp], dim=1)
        out = self.conv(images)
        out = self.dropout(out)
        out = self.resnet(out)
        out = out.view(out.size(0), -1)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, dropout=0.5):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        h = self.init_h(encoder_out)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            batch_embeds = embeddings[:batch_size_t, t, :]
            h, c = self.lstm(batch_embeds[:batch_size_t].float(), (h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
        return predictions, encoded_captions, decode_lengths, sort_ind
