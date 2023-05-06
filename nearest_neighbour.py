import numpy as np
from scipy.io import wavfile
import IPython.display as ipyd
import librosa.display
from scipy.spatial import distance as dist

from dynamic_programming import dp

# Audio
query_fn = "audio/hello1.wav"
f_s, x = wavfile.read(query_fn)

# Mel-scale spectrogram
n_fft = int(0.025*f_s)      # 25 ms
hop_length = int(0.01*f_s)  # 10 ms

# Mel-scale spectrogram
mel_spec_x = librosa.feature.melspectrogram(y=x / 1.0, sr=f_s, n_mels=40, n_fft=n_fft, hop_length=hop_length)
log_mel_spec_x = np.log(mel_spec_x)
x_seq = log_mel_spec_x.T

ipyd.Audio(rate=f_s, data=x)

audio_files = [
    "audio/hello2.wav", "audio/hello3.wav",
    "audio/bye.wav", "audio/cat.wav", "audio/goodbye.wav"
]
for neighbour_fn in audio_files:
    # Mel-scale spectrogram
    print("Reading:", neighbour_fn)
    f_s, y = wavfile.read(neighbour_fn)
    mel_spec_y = librosa.feature.melspectrogram(y=y / 1.0, sr=f_s, n_mels=40, n_fft=n_fft, hop_length=hop_length)
    log_mel_spec_y = np.log(mel_spec_y)
    y_seq = log_mel_spec_y.T

    dist_mat = dist.cdist(x_seq, y_seq, "cosine")
    path, cost_mat = dp(dist_mat)
    print("Alignment cost: {:.4f}".format(cost_mat[-1, -1]))
    M = y_seq.shape[0]
    N = x_seq.shape[0]
    print(
        "Normalized alignment cost: {:.8f}".format(
            cost_mat[-1, -1] / (M + N))
    )
    print()