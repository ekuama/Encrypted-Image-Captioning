import json

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import io
from tqdm import tqdm

caps_3 = []
image_names = []


def caption_image_beam_search(encoder, decoder, phase_image_path, amp_image_path, word_map_, device, bs=3,
                              robust='noise_0.25'):
    k = bs
    vocab_size = len(word_map_)
    infinite_pred = False

    # For each image
    phase = io.loadmat(phase_image_path)['P']
    amp = io.loadmat(amp_image_path)['A']
    phase = torch.FloatTensor(phase.transpose(2, 0, 1)).to(device)
    amp = torch.FloatTensor(amp.transpose(2, 0, 1)).to(device)

    if robust == 'noise_0.25':
        transform1 = transforms.Compose([transforms.Normalize((0.4975, 0.4975, 0.4975), (0.2897, 0.2897, 0.2896))])
        transform2 = transforms.Compose([transforms.Normalize((0.1258, 0.1199, 0.1105), (0.0665, 0.0634, 0.0584))])
    elif robust == 'noise_0.5':
        transform1 = transforms.Compose([transforms.Normalize((0.4975, 0.4975, 0.4975), (0.2897, 0.2897, 0.2896))])
        transform2 = transforms.Compose([transforms.Normalize((0.1176, 0.1121, 0.1033), (0.0634, 0.0605, 0.0557))])
    elif robust == 'noise_1':
        transform1 = transforms.Compose([transforms.Normalize((0.4975, 0.4975, 0.4975), (0.2897, 0.2897, 0.2896))])
        transform2 = transforms.Compose([transforms.Normalize((0.1071, 0.1021, 0.0941), (0.0607, 0.0579, 0.0533))])
    elif robust == 'exclude_0.2':
        transform1 = transforms.Compose([transforms.Normalize((0.4980, 0.4980, 0.4980), (0.2590, 0.2590, 0.2590))])
        transform2 = transforms.Compose([transforms.Normalize((0.1102, 0.1051, 0.0968), (0.0848, 0.0809, 0.0745))])
    elif robust == 'exclude_0.4':
        transform1 = transforms.Compose([transforms.Normalize((0.4979, 0.4980, 0.4980), (0.2247, 0.2247, 0.2246))])
        transform2 = transforms.Compose([transforms.Normalize((0.0828, 0.0789, 0.0727), (0.0828, 0.0789, 0.0727))])
    else:
        transform1 = transforms.Compose([transforms.Normalize((0.4985, 0.4985, 0.4986), (0.1830, 0.1830, 0.1830))])
        transform2 = transforms.Compose([transforms.Normalize((0.0597, 0.0581, 0.0541), (0.0882, 0.0859, 0.0800))])

    phase_in = transform1(phase)
    amp_in = transform2(amp)

    # Encode
    phase_in = phase_in.unsqueeze(0)
    amp_in = amp_in.unsqueeze(0)
    encoder_out = encoder(phase_in, amp_in)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map_['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        cat_val = torch.cat([embeddings.double(), awe.double()], dim=1)
        h, c = decoder.lstm(cat_val.float(), (h.float(), c.float()))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map_['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds.long()[incomplete_inds]]
        c = c[prev_word_inds.long()[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds.long()[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            infinite_pred = True
            break
        step += 1

    if infinite_pred is not True:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        seq = seqs[0][:20]
        seq = [seq[i].item() for i in range(len(seq))]

    return seq


def caption_search(image_name, name, robust):
    word_map_file = 'data_files/WORDMAP_flickr8k_5_3.json'
    phase_img = 'testing/' + robust + '/Phase/' + image_name + '.jpg.mat'
    amp_img = 'testing/' + robust + '/Amp/' + image_name + '.jpg.mat'

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(word_map_file, 'r') as z:
        word_map_ = json.load(z)
    rev_word_map_ = {v: k for k, v in word_map_.items()}

    if name == "ResNet50":
        checkpoint = 'BEST_ResNet50_attend.pth.tar'
    elif name == "ResNet101":
        checkpoint = 'BEST_ResNet101_attend.pth.tar'
    else:
        checkpoint = 'BEST_ResNeXt101_attend.pth.tar'

    image_names.append(image_name)

    checkpoint = torch.load(checkpoint, map_location=device_)
    decoder_ = checkpoint['decoder']
    decoder_ = decoder_.to(device_)
    decoder_.eval()
    encoder_ = checkpoint['encoder']
    encoder_ = encoder_.to(device_)
    encoder_.eval()

    beam_size = 3
    seq = caption_image_beam_search(encoder_, decoder_, phase_img, amp_img, word_map_, device_, beam_size, robust)
    hypo_caps = [w for w in seq if w not in {word_map_['<start>'], word_map_['<end>'], word_map_['<pad>']}]
    words = ' '.join(rev_word_map_[f] for f in hypo_caps)
    caps_3.append(words)


with open('dataset_flickr8k.json', 'r') as j:
    data = json.load(j)
test_image_paths = []
for img in data['images']:
    path = img['filename']
    path = path[:-4]
    if img['split'] in {'test'}:
        test_image_paths.append(path)
cnn_names = ['ResNet50', 'ResNet101', 'ResNeXt101']
robustness = ['noise_0.25', 'noise_0.5', 'noise_1', 'exclude_0.2', 'exclude_0.4', 'exclude_0.6']
for r in robustness:
    for cnn in cnn_names:
        print('-------------------------' + cnn + '-----------------------------------------')
        for o, fname in enumerate(tqdm(test_image_paths)):
            caption_search(fname, cnn, r)

        name_dict = {'Name': image_names, 'Beam 3': caps_3}
        caption_nic = pd.DataFrame(name_dict)
        caption_nic.to_csv('results_csv/' + cnn + '_caption_attend_' + r + '.csv', index=False)

        caps_3 = []
        image_names = []
