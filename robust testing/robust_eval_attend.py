import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm import tqdm

from data.datasets_robust_test import CaptionDataset
from attend.utils_attend import get_eval_score


def evaluate(beam_size_, data_folder, data_name, device, encoder, decoder, word_map, vocab_size, rev_word_map, robust):
    loader = torch.utils.data.DataLoader(CaptionDataset(data_folder, data_name, robust),
                                         batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    references = list()
    hypotheses = list()

    # For each image
    with torch.no_grad():
        for i, (phase, amp, caps, caplens, all_caps) in enumerate(tqdm(loader)):

            k = beam_size_
            infinite_pred = False

            phase = phase.to(device)
            amp = amp.to(device)

            # Encode
            encoder_out = encoder(phase, amp)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            # We'll treat the problem as having a batch size of k
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
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

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
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

            # References
            img_caps = all_caps[0].tolist()
            img_captions = list(
                map(lambda c_: [w for w in c_ if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            i_c = list(map(lambda m: [rev_word_map[j_] for j_ in m], img_captions))
            references.append(i_c)

            # Hypotheses
            hypo_caps = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append([rev_word_map[f] for f in hypo_caps])

            assert len(references) == len(hypotheses)

    metrics = get_eval_score(references, hypotheses)

    return metrics


if __name__ == '__main__':
    # Parameters
    data_folder_ = 'data_files'
    data_name_ = 'flickr8k_5_3'  # base name shared by data files
    word_map_file_ = 'data_files/WORDMAP_flickr8k_5_3.json'
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Load word map (word2ix)
    with open(word_map_file_, 'r') as j:
        word_map_ = json.load(j)
    rev_word_map_ = {v: k for k, v in word_map_.items()}
    vocab_size_ = len(word_map_)

    checkpoints = ['BEST_ResNet50_attend.pth.tar', 'BEST_ResNet101_attend.pth.tar', 'BEST_ResNeXt101_attend.pth.tar']
    robustness = ['noise_0.25', 'noise_0.5', 'noise_1', 'exclude_0.2', 'exclude_0.4', 'exclude_0.6']

    for r in robustness:
        print('-----------------------' + r + '----------------------------')
        for checkpoint in checkpoints:
            print(checkpoint)

            # Load model
            checkpoint = torch.load('models/' + checkpoint, map_location=device_)
            decoder_ = checkpoint['decoder']
            decoder_ = decoder_.to(device_)
            decoder_.eval()
            encoder_ = checkpoint['encoder']
            encoder_ = encoder_.to(device_)
            encoder_.eval()

            beam_size = 3
            eval_metrics = evaluate(beam_size, data_folder_, data_name_, device_, encoder_, decoder_,
                                    word_map_, vocab_size_, rev_word_map_, r)
            print("Beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
                  (beam_size, eval_metrics["Bleu_1"], eval_metrics["Bleu_2"], eval_metrics["Bleu_3"],
                   eval_metrics["Bleu_4"], eval_metrics["METEOR"], eval_metrics["ROUGE_L"], eval_metrics["CIDEr"]))
