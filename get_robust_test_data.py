import json
import os

from scipy import io

with open('dataset_flickr8k.json', 'r') as j:
    data = json.load(j)
test_image_paths = []
for img in data['images']:
    path = img['filename']
    path = path[:-4]
    if img['split'] in {'test'}:
        test_image_paths.append(path)

robustness = ['Noise_0.25', 'Noise_0.5', 'Noise_1', 'Exclude_20', 'Exclude_40', 'Exclude_60']

for fname in test_image_paths:
    fname = fname + '.jpg.mat'
    for r in robustness:
        amp_root = 'image_files' + r + '/Amp'
        phase_root = 'image_files' + r + '/Phase'
        amp_path = os.path.join(amp_root, fname)
        phase_path = os.path.join(phase_root, fname)
        amp_img = io.loadmat(amp_path)['A']
        phase_img = io.loadmat(phase_path)['P']
        amp_target = 'testing/' + r + '/Amp'
        phase_target = 'testing/' + r + '/Phase'
        if not os.path.isdir(amp_target):
            os.makedirs(amp_target)
        if not os.path.isdir(phase_target):
            os.makedirs(phase_target)
        target_amp = os.path.join(amp_target, fname)
        mdict = {'A': amp_img}
        io.savemat(target_amp, mdict=mdict)
        target_phase = os.path.join(phase_target, fname)
        mdict = {'P': phase_img}
        io.savemat(target_phase, mdict=mdict)
