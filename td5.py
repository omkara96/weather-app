import struct
import wave
import numpy as np
from reedsolo import RSCodec, ReedSolomonError


def fast_cross_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """FFT-accelerated 1D cross-correlation (mode='valid') in pure NumPy."""
    n_sig = len(signal)
    n_tpl = len(template)
    n_fft = 1 << int(np.ceil(np.log2(n_sig + n_tpl - 1)))

    S = np.fft.rfft(signal, n=n_fft)
    T = np.fft.rfft(template, n=n_fft)

    corr_full = np.fft.irfft(S * np.conj(T), n=n_fft)
    return corr_full[: n_sig - n_tpl + 1]


def find_first_prominent_peak(corr: np.ndarray, sr: int) -> int:
    """Finds the apex of the FIRST prominent correlation peak."""
    max_val = np.max(corr)
    if max_val <= 0:
        return 0

    threshold = 0.35 * max_val
    above_thresh = np.where(corr > threshold)[0]

    if len(above_thresh) == 0:
        return int(np.argmax(corr))

    first_idx = above_thresh[0]
    search_window = min(len(corr), first_idx + int(sr * 0.1))
    local_peak_offset = np.argmax(corr[first_idx:search_window])

    return int(first_idx + local_peak_offset)


def find_last_prominent_peak(corr: np.ndarray, sr: int) -> int:
    """Finds the apex of the LAST prominent correlation peak."""
    max_val = np.max(corr)
    if max_val <= 0:
        return 0

    threshold = 0.35 * max_val
    above_thresh = np.where(corr > threshold)[0]

    if len(above_thresh) == 0:
        return int(np.argmax(corr))

    last_idx = above_thresh[-1]
    search_start = max(0, last_idx - int(sr * 0.1))
    local_peak_offset = np.argmax(corr[search_start : last_idx + 1])

    return int(search_start + local_peak_offset)


def decode_fast_wideband_ofdm(
    wav_path: str = "acoustic_1.6mb.wav",
    output_path: str = "recovered_myfile.7z",
    sample_rate: int = 44100,
    n_fft: int = 1024,
    cp_len: int = 64,
    ecc_bytes: int = 32,
    block_size: int = 223,
):
    # 1. Load Audio
    print(f"Reading '{wav_path}'...")
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        n_frames = w.getnframes()
        raw_bytes = w.readframes(n_frames)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

        if n_channels > 1:
            print(f"Detected {n_channels}-channel stereo recording. Downmixing to mono...")
            audio = audio.reshape(-1, n_channels).mean(axis=1)

    duration_sec = len(audio) / sr
    print(f"Audio loaded: {len(audio):,} samples ({duration_sec / 60:.2f} minutes).")

    # 2. Start Beacon Sync (1 kHz -> 18 kHz Chirp)
    print("Locating Start Beacon (1-18 kHz Wideband Matched Filter)...")
    t_chirp = np.arange(int(sr * 0.5)) / float(sr)
    start_ref = np.sin(2 * np.pi * (1000.0 + (18000.0 - 1000.0) * t_chirp / 2.0) * t_chirp)
    start_ref = (start_ref * np.hanning(len(start_ref))).astype(np.float32)

    corr_start = fast_cross_correlation(audio, start_ref)
    sync_idx = find_first_prominent_peak(corr_start, sr)
    payload_start = sync_idx + len(start_ref)

    print(f"-> Start Beacon locked at sample {sync_idx} ({sync_idx / sr:.2f}s).")

    # 3. End Beacon Sync (18 kHz -> 1 kHz Chirp)
    print("Locating End Beacon...")
    end_ref = np.sin(2 * np.pi * (18000.0 + (1000.0 - 18000.0) * t_chirp / 2.0) * t_chirp)
    end_ref = (end_ref * np.hanning(len(end_ref))).astype(np.float32)

    corr_end = fast_cross_correlation(audio[payload_start:], end_ref)
    payload_length_samples = find_last_prominent_peak(corr_end, sr)

    print(f"-> End Beacon locked at sample {payload_start + payload_length_samples}.")
    print(
        f"-> Extracted Payload Region: {payload_length_samples:,} samples ({(payload_length_samples / sr) / 60:.2f} minutes)."
    )

    payload = audio[payload_start : payload_start + payload_length_samples]

    # 4. Extract OFDM Frames
    frame_len = n_fft + cp_len
    num_frames = len(payload) // frame_len
    payload = payload[: num_frames * frame_len]

    print(f"Processing {num_frames:,} wideband OFDM frames...")
    frames = payload.reshape((num_frames, frame_len))
    frames_no_cp = frames[:, cp_len:]

    # 5. FFT Analysis & Active Carrier Filtering
    spectrums = np.fft.rfft(frames_no_cp, axis=1)

    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    active_carriers = np.where((freq_bins >= 1000) & (freq_bins <= 18000))[0]
    rx_active = spectrums[:, active_carriers].flatten()

    # 6. 16-QAM Equalization & Demodulation
    qam16_constellation = np.array([
        -3+3j, -3+1j, -3-1j, -3-3j,
        -1+3j, -1+1j, -1-1j, -1-3j,
         1+3j,  1+1j,  1-1j,  1-3j,
         3+3j,  3+1j,  3-1j,  3-3j
    ], dtype=np.complex64) / np.sqrt(10.0)

    # Normalize power across received subcarriers
    rms_power = np.sqrt(np.mean(np.abs(rx_active) ** 2))
    if rms_power > 0:
        rx_equalized = rx_active / rms_power
    else:
        rx_equalized = rx_active

    print("Demodulating 16-QAM symbols...")
    # Compute minimum Euclidean distance to reference constellation points
    distances = np.abs(rx_equalized[:, None] - qam16_constellation[None, :])
    detected_indices = np.argmin(distances, axis=1).astype(np.uint8)

    # Reconstruct bytes from 4-bit nibbles
    # Combine every pair of nibbles (high_nibble, low_nibble) -> byte
    if len(detected_indices) % 2 != 0:
        detected_indices = detected_indices[:-1]

    high_nibbles = detected_indices[0::2]
    low_nibbles = detected_indices[1::2]
    rx_bytes = bytes(((high_nibbles << 4) | low_nibbles).astype(np.uint8))

    # 7. Reed-Solomon Error Correction
    print("Running Reed-Solomon error correction...")
    rs = RSCodec(ecc_bytes)
    total_rs_block_size = block_size + ecc_bytes

    decoded_payload = bytearray()
    corrected_blocks = 0
    failed_blocks = 0

    for i in range(0, len(rx_bytes), total_rs_block_size):
        chunk = rx_bytes[i : i + total_rs_block_size]
        if len(chunk) < total_rs_block_size:
            break

        try:
            clean = rs.decode(chunk)
            clean_block = clean[0] if isinstance(clean, tuple) else clean
            decoded_payload.extend(clean_block)
            corrected_blocks += 1
        except ReedSolomonError:
            failed_blocks += 1

    print(f"RS Summary: {corrected_blocks:,} blocks decoded successfully | {failed_blocks:,} failed blocks")

    if len(decoded_payload) < 4:
        raise ValueError("Error: Decoded payload is too short or recording was corrupted.")

    # Unpack 4-byte big-endian file size header
    orig_len = struct.unpack(">I", decoded_payload[:4])[0]
    print(f"Target File Size: {orig_len:,} bytes ({orig_len / (1024 * 1024):.2f} MB)")

    zip_bytes = bytes(decoded_payload[4 : 4 + orig_len])
    with open(output_path, "wb") as f:
        f.write(zip_bytes)

    print(f"\nSuccessfully restored file -> '{output_path}' ({len(zip_bytes):,} bytes)")


if __name__ == "__main__":
    decode_fast_wideband_ofdm(
        wav_path="abc2.wav",
        output_path="myfile.zip",
    )
