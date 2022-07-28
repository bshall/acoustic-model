import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm

import torchaudio
from torchaudio.functional import resample

from acoustic.utils import LogMelSpectrogram


melspectrogram = LogMelSpectrogram()


def process_wav(in_path, out_path):
    wav, sr = torchaudio.load(in_path)
    wav = resample(wav, sr, 16000)

    logmel = melspectrogram(wav.unsqueeze(0))

    np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())
    return out_path, logmel.shape[-1]


def preprocess_dataset(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    print(f"Extracting features for {args.in_dir}")
    for in_path in args.in_dir.rglob("*.wav"):
        relative_path = in_path.relative_to(args.in_dir)
        out_path = args.out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        futures.append(executor.submit(process_wav, in_path, out_path))

    results = [future.result() for future in tqdm(futures)]

    lengths = {path.stem: length for path, length in results}
    frames = sum(lengths.values())
    frame_shift_ms = 160 / 16000
    hours = frames * frame_shift_ms / 3600
    print(f"Wrote {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mel-spectrograms for an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    preprocess_dataset(args)
