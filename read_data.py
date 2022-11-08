import json
import os
from collections import Counter
from random import seed, choice, sample

import h5py
from scipy import io
from tqdm import tqdm


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=40, encode='P'):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if encode == 'P':
        name = 'phase'
    elif encode == 'A':
        name = 'amp'
    elif encode == 'R':
        name = 'real'
    elif encode == 'J':
        name = 'imag'
    else:
        name = 'resize'
    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])
        path = path + '.mat'

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_' + str(min_word_freq)

    # Save word map to a JSON
    if encode == 'P':
        with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, name + '_' + split + '_IMAGES_' + base_filename + '.hdf5'),
                       'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 256, 256, 3), dtype='float32')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = io.loadmat(path)[encode]
                img = img.transpose(1, 2, 0)
                assert img.shape == (256, 256, 3)

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            if encode == 'P':
                # Save encoded captions and their lengths to JSON files
                with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                    json.dump(enc_captions, j)

                with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                    json.dump(caplens, j)


create_input_files('flickr8k', 'dataset_flickr8k.json', 'image_files/Phase',
                   5, 3, 'data_files', max_len=40, encode='P')
create_input_files('flickr8k', 'dataset_flickr8k.json', 'image_files/Amp',
                   5, 3, 'data_files', max_len=40, encode='A')

"""
For real and imaginary data
"""
# create_input_files('flickr8k', 'dataset_flickr8k.json', 'image_files/Real',
#                    5, 3, 'data_files', max_len=40, encode='R')
# create_input_files('flickr8k', 'dataset_flickr8k.json', 'image_files/Imag',
#                    5, 3, 'data_files', max_len=40, encode='J')
