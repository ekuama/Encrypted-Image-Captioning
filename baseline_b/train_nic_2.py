import argparse
import json
import os
import random
import time
import h5py
import numpy as np

import torch
import torch.optim
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data.datasets import CaptionDataset
from model_nic_2 import EncoderA, Decoder
from utils_nic_2 import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint

random_seed = random.randint(1, 10000)  # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
print(random_seed)


def main(arg):
    device = torch.device("cuda")
    best_loss = 100.00

    # Read word map
    word_map_file = os.path.join(arg.data_folder, 'WORDMAP_' + arg.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

        # Initialize / load checkpoint
        if arg.checkpoint is None:
            decoder = Decoder(embed_dim=arg.emb_dim, decoder_dim=arg.decoder_dim,
                              vocab_size=len(word_map), dropout=arg.dropout)
            encoder = EncoderA(arg.cnn_name)
            model_optimizer = torch.optim.Adam([{'params': decoder.parameters()},
                                                {'params': encoder.parameters()}],
                                               lr=arg.decoder_lr, weight_decay=1e-4)
            best_bleu4 = arg.best_bleu4
            epochs_since_improvement = arg.epochs_since_improvement
            start_epoch = arg.start_epoch

        else:
            ckpt = torch.load(arg.checkpoint, map_location=torch.device(device))
            start_epoch = ckpt['epoch'] + 1
            epochs_since_improvement = ckpt['epochs_since_improvement']
            best_bleu4 = ckpt['recent']
            decoder = ckpt['decoder']
            model_optimizer = ckpt['model_optimizer']
            encoder = ckpt['encoder']

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom data loaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(arg.data_folder, arg.data_name, 'TRAIN', False), batch_size=arg.batch_size, shuffle=True,
        num_workers=arg.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(arg.data_folder, arg.data_name, 'VAL', False), batch_size=arg.batch_size, shuffle=True,
        num_workers=arg.workers, pin_memory=True)

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    val_bleu4 = []

    # Epochs
    for epoch in range(start_epoch, arg.epochs):

        if epochs_since_improvement != 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(model_optimizer, 0.8)

        # One epoch's training
        t_loss, t_acc = train(arg=arg, train_loader=train_loader, encoder=encoder, decoder=decoder,
                              criterion=criterion,  model_optimizer=model_optimizer, epoch=epoch, device=device)

        # One epoch's validation
        recent_bleu4, v_loss, v_acc = validate(arg=arg, val_loader=val_loader, encoder=encoder, decoder=decoder,
                                               criterion=criterion, device=device, word_map=word_map)

        train_losses.append(t_loss)
        train_acc.append(t_acc)
        val_losses.append(v_loss)
        val_acc.append(v_acc)
        val_bleu4.append(recent_bleu4)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        is_less = v_loss < best_loss
        best_loss = min(v_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(arg.cnn_name, epoch, epochs_since_improvement, encoder, decoder,
                        model_optimizer, recent_bleu4, is_best, is_less)

    with h5py.File(os.path.join('models/' + arg.cnn_name + '_nic_2.hdf5'), 'a') as h:
        h.attrs['train_loss'] = train_losses
        h.attrs['train_acc'] = train_acc
        h.attrs['val_losses'] = val_losses
        h.attrs['val_acc'] = val_acc
        h.attrs['val_bleu4'] = val_bleu4


def train(arg, train_loader, encoder, decoder, criterion, model_optimizer, epoch, device):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (phase, amp, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        phase = phase.to(device)
        amp = amp.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(phase, amp)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        model_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if arg.grad_clip is not None:
            clip_gradient(model_optimizer, arg.grad_clip)

        # Update weights
        model_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % arg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch + 1, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    print('Epoch: [{0}] LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(epoch + 1, loss=losses,
                                                                                         top5=top5accs))
    return losses.avg, top5accs.avg


def validate(arg, val_loader, encoder, decoder, criterion, device, word_map):
    decoder.eval()  # eval mode (no dropout or batch norm)
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (phase, amp, caps, caplens, all_caps) in enumerate(val_loader):

            # Move to device, if available
            phase = phase.to(device)
            amp = amp.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            imgs = encoder(phase, amp)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % arg.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # References
            all_caps = all_caps[sort_ind]  # because images were sorted in the decoder
            for l in range(all_caps.shape[0]):
                img_caps = all_caps[l].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for l, p in enumerate(preds):
                temp_preds.append(preds[l][:decode_lengths[l]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate scores
        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU4 - {bleu}\n'.format(
            loss=losses, top5=top5accs, bleu=bleu4))

    return bleu4, losses.avg, top5accs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    # Data parameters
    parser.add_argument('--data_folder', type=str, help='path to data information',
                        default='data_files')
    parser.add_argument('--data_name', type=str, help='dataset name + _5_3',
                        default='flickr8k_5_3')

    # Model parameters
    parser.add_argument('--emb_dim', type=int, help='dimension of word embeddings', default=512)
    parser.add_argument('--decoder_dim', type=int, help='dimension of decoder RNN', default=512)
    parser.add_argument('--dropout', type=float, help='rate of dropout', default=0.5)
    parser.add_argument('--cnn_name', type=str, default='ResNet50')
    parser.add_argument('--benchmark', type=bool, default=True,
                        help='set to true only if inputs to model are fixed size; otherwise lot of computational '
                             'overhead')

    # Training parameters
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--epochs_since_improvement', type=int, default=0,
                        help='keeps track of number of epochs since there is been an improvement in validation BLEU')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0,
                        help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', type=float, default=4e-4,
                        help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=4e-4,
                        help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=.5,
                        help='clip gradients at an absolute value of')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention, as in the paper')
    parser.add_argument('--best_bleu4', type=int, default=0,
                        help='BLEU-4 score right now')
    parser.add_argument('--print_freq', type=float, default=100,
                        help='print training/validation stats every __ batches')
    parser.add_argument('--fine_tune_encoder', type=bool, default=False,
                        help='keeps track of number of epochs since there is been an improvement in validation BLEU')
    parser.add_argument('--checkpoint', default=None)

    args = parser.parse_args()

    print(args)

    main(args)
