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
            modules = list(resnet.children())[1:-2]
            self.resnet = nn.Sequential(*modules)
        elif self.name == 'ResNeXt101':
            self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet = torchvision.models.resnext101_32x8d(pretrained=True)
            modules = list(resnet.children())[1:-2]
            self.resnet = nn.Sequential(*modules)
        else:
            self.conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet = torchvision.models.resnet50(pretrained=True)
            modules = list(resnet.children())[1:-2]
            self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, phase, amp):
        images = torch.cat([phase, amp], dim=1)
        out = self.conv(images)
        out = self.dropout(out)
        out = self.resnet(out)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (1, batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (1, batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (1, batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, dropout=0.5):

        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(self.encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(embed_dim + self.encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
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
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (1, batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (1, batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        # (batch_size, max(decode_lengths), vocab_size)
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        # (batch_size, max(decode_lengths), num_pixels)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        # batch_embeds = embeddings[:, 0, :]
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
