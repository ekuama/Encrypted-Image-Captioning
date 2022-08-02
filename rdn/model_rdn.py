import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_A(nn.Module):
    def __init__(self, name, encoded_image_size=14):
        super(Encoder_A, self).__init__()
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, phase, amp):
        images = torch.cat([phase, amp], dim=1)
        out = self.conv(images)
        out = self.dropout(out)
        out = self.resnet(out)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.adaptive_pool_(out)
        out = out.view(out.size(0), -1)
        return out


class Encoder_B(nn.Module):

    def __init__(self, name):
        super(Encoder_B, self).__init__()
        self.name = name
        if self.name == 'ResNet101':
            resnet = torchvision.models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
        elif self.name == 'ResNet50':
            resnet = torchvision.models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
        else:
            resnet = torchvision.models.resnext101_32x8d(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        out = self.resnet(images)
        out = out.view(out.size(0), -1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[4:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention_Visual(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        super(Attention_Visual, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    def __init__(self, name, attention_dim, embed_dim, decoder_dim, vocab_size, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention_Visual(self.encoder_dim, decoder_dim, attention_dim)
        self.f_beta = nn.Linear(decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm1 = nn.LSTMCell(embed_dim + self.encoder_dim + decoder_dim, decoder_dim, bias=True)
        self.lstm2 = nn.LSTMCell(self.encoder_dim + decoder_dim, decoder_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.ram_fc = nn.Linear(decoder_dim, vocab_size)
        self.rpm_fc = nn.Linear(decoder_dim, 1, bias=False)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.ram_fc.bias.data.fill_(0)
        self.ram_fc.weight.data.uniform_(-0.1, 0.1)
        self.rpm_fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        h = torch.zeros(encoder_out.shape[0], self.decoder_dim).to(device)
        c = torch.zeros(encoder_out.shape[0], self.decoder_dim).to(device)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(encoder_out)
        h2, c2 = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        relative_pos = torch.zeros(batch_size, max(decode_lengths))
        for h in range(len(decode_lengths)):
            for l in range(decode_lengths[h]):
                if (l+1)/decode_lengths[h] > 1.000:
                    relative_pos[h, l] = 0.0000
                else:
                    relative_pos[h, l] = (l+1)/decode_lengths[h]

        # Create tensors to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predicted_pos = torch.zeros(batch_size, max(decode_lengths))
        mean_encoder_out = encoder_out.mean(dim=1)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            batch_embeds = embeddings[:batch_size_t, t, :]
            cat_val = torch.cat([batch_embeds.double(), h2[:batch_size_t].double(),
                                 mean_encoder_out[:batch_size_t].double()], dim=1)
            h1, c1 = self.lstm1(cat_val.float(), (h1[:batch_size_t].float(), c1[:batch_size_t].float()))
            att_vis = self.attention(encoder_out[:batch_size_t], h1)
            gate = self.sigmoid(self.f_beta(h1[:batch_size_t]))
            att_vis = gate * att_vis
            cat_val2 = torch.cat([att_vis[:batch_size_t].double(), h1[:batch_size_t].double()], dim=1)
            h2, c2 = self.lstm2(cat_val2.float(), (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.ram_fc(self.dropout(h2))
            rpm_out = self.sigmoid(self.rpm_fc(self.dropout(h2)))
            predictions[:batch_size_t, t, :] = preds[:batch_size_t]
            predicted_pos[:batch_size_t, t] = rpm_out[:batch_size_t].squeeze(1)

        return predictions, encoded_captions, decode_lengths, sort_ind, relative_pos, predicted_pos
