import re
import argparse
from string import punctuation
import torch
import yaml
import numpy as np
import os
import json

import librosa
import pyworld as pw
# import audio as Audio
import processing as Audio

from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from utils.dataset import BatchInferenceDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HOMEDIR = os.getcwd()


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def get_audio(preprocess_config, wav_path):

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        hop_length,
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        sampling_rate,
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"][2:] + stats["energy"][2:]
        pitch_mean, pitch_std, energy_mean, energy_std = stats

    # Read and trim wav files
    wav, _ = librosa.load(wav_path)

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        frame_period=hop_length / sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav.astype(np.float32), STFT)

    # Normalize Variance
    pitch = (pitch - pitch_mean) / pitch_std
    energy = (energy - energy_mean) / energy_std

    mels = mel_spectrogram.T[None]
    mel_lens = np.array([len(mels[0])])

    mel_spectrogram = mel_spectrogram.astype(np.float32)
    energy = energy.astype(np.float32)

    return mels, mel_lens, (mel_spectrogram, pitch, energy)


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-1]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )



if __name__=="__main__":

    # preprocess_config
    process_config_path = os.path.join(HOMEDIR, "config", "LibriTTS", "preprocess.yaml")
    model_config = os.path.join(HOMEDIR, "config", "LibriTTS", "model.yaml")
    train_config = os.path.join(HOMEDIR, "config", "LibriTTS", "train.yaml")


    # Read config
    preprocess_config = yaml.load(
        open(process_config_path, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config, train_config)

    # Reference Audio (recruiter audio sample path)
    ref_audio = os.path.join(HOMEDIR, 'data', 'audio.wav')

    # Restore step used for loading model checkpoint
    restore_step = 200000

    # Mode: {single, batch}
    mode = "single"
    source = None
    if mode == "batch":
        # Source of text files to transcribe \
        # (required if transcription is required in batch mode)
        dataset = BatchInferenceDataset(source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn
        )

    text = "This is sample for generating speech from given text"
    if mode == "single":
        # process the text input for transcription
        ids = raw_texts = [text[:100]]
        texts = np.array([preprocess_english(text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        mels, mel_lens, ref_info = get_audio(preprocess_config, ref_audio)
        batchs = [(["_".join([os.path.basename(ref_audio).strip(".wav"), id]) for id in ids], \
            raw_texts, None, texts, text_lens, max(text_lens), mels, mel_lens, max(mel_lens), [ref_info])]

    pitch_control=1.0
    energy_control=5.0
    duration_control=1.0
    control_values = pitch_control, energy_control, duration_control

    
    args = {
        'restore_step': restore_step
    }
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    synthesize(model, args['restore_step'], configs, vocoder, batchs, control_values)