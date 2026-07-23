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


def decode_fast_acoustic_ofdm_to_zip(
    wav_path: str,
    output_zip_path: str = "recovered_myfile.zip",
    sample_rate: int = 44100,
    n_fft: int = 1024,
    cp_len: int = 128,
    ecc_bytes: int = 32,
    block_size: int = 223,
):
    print(f"Reading '{wav_path}'...")
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n_frames = w.getnframes()
        audio = np.frombuffer(w.readframes(n_frames), dtype=np.int16).astype(np.float32)

    duration_sec = len(audio) / sr
    print(f"Audio loaded: {len(audio):,} samples ({duration_sec / 60:.2f} minutes).")

    # 1. Instant Start Preamble Lock (FFT Matched Filter)
    print("Locating Start Beacon (FFT Matched Filtering)...")
    t_chirp = np.arange(int(sr * 0.8)) / float(sr)
    ref_chirp = np.sin(2 * np.pi * (1200.0 + (13800.0 - 1200.0) * t_chirp / 2.0) * t_chirp)
    ref_chirp = (ref_chirp * np.hanning(len(ref_chirp))).astype(np.float32)

    corr_start = fast_cross_correlation(audio, ref_chirp)
    sync_idx = int(np.argmax(corr_start))
    payload_start = sync_idx + len(ref_chirp)

    print(f"-> Start Beacon locked at sample {sync_idx} ({sync_idx / sr:.2f}s).")

    # 2. Instant End Beacon Lock
    print("Locating End Beacon...")
    ref_end_chirp = np.sin(2 * np.pi * (13800.0 + (1200.0 - 13800.0) * t_chirp / 2.0) * t_chirp)
    ref_end_chirp = (ref_end_chirp * np.hanning(len(ref_end_chirp))).astype(np.float32)

    corr_end = fast_cross_correlation(audio[payload_start:], ref_end_chirp)
    payload_length_samples = int(np.argmax(corr_end))

    print(f"-> End Beacon locked at sample {payload_start + payload_length_samples}.")
    print(f"-> Extracted Payload Region: {payload_length_samples:,} samples.")

    payload = audio[payload_start : payload_start + payload_length_samples]

    # 3. Setup Subcarrier Indexing
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    active_carriers = np.where((freq_bins >= 1200) & (freq_bins <= 13800))[0]
    num_active = len(active_carriers)

    pilot_indices_in_active = np.arange(0, num_active, 8)
    data_indices_in_active = np.delete(np.arange(num_active), pilot_indices_in_active)

    # 4. Demux OFDM Frames
    frame_len = n_fft + cp_len
    num_frames = len(payload) // frame_len
    payload = payload[: num_frames * frame_len]

    print(f"Processing {num_frames:,} OFDM frames...")
    frames = payload.reshape((num_frames, frame_len))
    frames_no_cp = frames[:, cp_len:]

    # 5. FFT & Equalization
    spectrums = np.fft.rfft(frames_no_cp, axis=1)
    rx_active = spectrums[:, active_carriers]

    rx_pilots = rx_active[:, pilot_indices_in_active]
    tx_pilots = 1.0 + 0.0j
    H_pilots = rx_pilots / tx_pilots

    equalized_active = np.zeros_like(rx_active)
    active_range = np.arange(num_active)

    for i in range(num_frames):
        H_real = np.interp(active_range, pilot_indices_in_active, np.real(H_pilots[i]))
        H_imag = np.interp(active_range, pilot_indices_in_active, np.imag(H_pilots[i]))
        H_interp = H_real + 1j * H_imag
        equalized_active[i] = rx_active[i] / H_interp

    equalized_data = equalized_active[:, data_indices_in_active]
    flattened_data = equalized_data.flatten()

    # 6. Demodulate 16-QAM
    print("Demodulating 16-QAM symbols...")
    qam16_table = np.array(
        [
            -3 + 3j, -3 + 1j, -3 - 1j, -3 - 3j,
            -1 + 3j, -1 + 1j, -1 - 1j, -1 - 3j,
             1 + 3j,  1 + 1j,  1 - 1j,  1 - 3j,
             3 + 3j,  3 + 1j,  3 - 1j,  3 - 3j,
        ],
        dtype=np.complex64,
    ) / np.sqrt(10.0)

    dists = np.abs(flattened_data[:, np.newaxis] - qam16_table[np.newaxis, :])
    rx_symbols = np.argmin(dists, axis=1)

    if len(rx_symbols) % 2 != 0:
        rx_symbols = rx_symbols[:-1]

    highs = rx_symbols[0::2]
    lows = rx_symbols[1::2]
    rx_bytes = bytes(((highs << 4) | lows).astype(np.uint8))

    # 7. Reed-Solomon Decoding
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

    orig_len = struct.unpack(">I", decoded_payload[:4])[0]
    print(f"Target ZIP File Size: {orig_len:,} bytes")

    zip_bytes = bytes(decoded_payload[4 : 4 + orig_len])
    with open(output_zip_path, "wb") as f:
        f.write(zip_bytes)

    print(f"\nSuccessfully restored file -> '{output_zip_path}' ({len(zip_bytes):,} bytes)")


if __name__ == "__main__":
    decode_fast_acoustic_ofdm_to_zip(
        wav_path="acoustic_archive.wav",
        output_zip_path="recovered_myfile.zip",
    )