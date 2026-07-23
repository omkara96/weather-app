import wave
import struct
import numpy as np
from reedsolo import RSCodec

def encode_large_file_fast_ofdm(
    input_file_path: str,
    output_wav_path: str = "acoustic_1.6mb.wav",
    sample_rate: int = 44100,
    n_fft: int = 1024,
    cp_len: int = 64,       # Shortened CP for higher throughput
    ecc_bytes: int = 32,
    block_size: int = 223
):
    with open(input_file_path, "rb") as f:
        raw_bytes = f.read()

    print(f"Input File Size: {len(raw_bytes):,} bytes ({len(raw_bytes)/(1024*1024):.2f} MB)")

    # 1. Header & RS Padding
    header = struct.pack(">I", len(raw_bytes))
    payload = header + raw_bytes

    pad_rs = (block_size - (len(payload) % block_size)) % block_size
    payload_padded = payload + b"\x00" * pad_rs

    # 2. Reed-Solomon Encoding
    print("Applying Reed-Solomon protection...")
    rs = RSCodec(ecc_bytes)
    protected_bytes = bytearray()
    for i in range(0, len(payload_padded), block_size):
        chunk = payload_padded[i : i + block_size]
        protected_bytes.extend(rs.encode(chunk))

    # 3. 16-QAM Symbol Mapping (4 bits / symbol)
    byte_array = np.frombuffer(protected_bytes, dtype=np.uint8)
    nibbles_high = (byte_array >> 4) & 0x0F
    nibbles_low = byte_array & 0x0F
    sym_indices = np.column_stack((nibbles_high, nibbles_low)).flatten()

    qam16_constellation = np.array([
        -3+3j, -3+1j, -3-1j, -3-3j,
        -1+3j, -1+1j, -1-1j, -1-3j,
         1+3j,  1+1j,  1-1j,  1-3j,
         3+3j,  3+1j,  3-1j,  3-3j
    ], dtype=np.complex64) / np.sqrt(10.0)

    qam_symbols = qam16_constellation[sym_indices]

    # 4. Wideband Subcarrier Indexing (1 kHz to 18 kHz)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    active_carriers = np.where((freq_bins >= 1000) & (freq_bins <= 18000))[0]
    num_active = len(active_carriers)

    total_frames = int(np.ceil(len(qam_symbols) / num_active))
    pad_syms = (total_frames * num_active) - len(qam_symbols)
    qam_symbols_padded = np.pad(qam_symbols, (0, pad_syms), mode='constant')
    symbol_matrix = qam_symbols_padded.reshape((total_frames, num_active))

    # 5. IFFT + Cyclic Prefix
    spectrum = np.zeros((total_frames, n_fft // 2 + 1), dtype=np.complex64)
    spectrum[:, active_carriers] = symbol_matrix
    time_frames = np.fft.irfft(spectrum, n=n_fft, axis=1)

    cp = time_frames[:, -cp_len:]
    frames_with_cp = np.hstack([cp, time_frames])
    payload_signal = frames_with_cp.flatten()

    # 6. Synchronization Beacons
    t_chirp = np.arange(int(sample_rate * 0.5)) / float(sample_rate)
    start_chirp = np.sin(2 * np.pi * (1000.0 + (18000.0 - 1000.0) * t_chirp / 2.0) * t_chirp)
    start_chirp = (start_chirp * np.hanning(len(start_chirp))).astype(np.float32)

    end_chirp = np.sin(2 * np.pi * (18000.0 + (1000.0 - 18000.0) * t_chirp / 2.0) * t_chirp)
    end_chirp = (end_chirp * np.hanning(len(end_chirp))).astype(np.float32)

    silence = np.zeros(int(sample_rate * 0.2))
    full_signal = np.concatenate([silence, start_chirp, payload_signal, end_chirp, silence])

    # 7. Write WAV Output
    max_val = np.max(np.abs(full_signal))
    scaled = (full_signal / max_val) * 28000.0
    pcm_data = scaled.astype(np.int16)

    with wave.open(output_wav_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data.tobytes())

    duration_min = (len(pcm_data) / sample_rate) / 60.0
    print(f"\nGenerated '{output_wav_path}' -> Total Duration: {duration_min:.2f} minutes")

if __name__ == "__main__":
    encode_large_file_fast_ofdm("myfile.zip", "acoustic_1.6mb.wav")

