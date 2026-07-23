import struct
import wave
import numpy as np
from reedsolo import RSCodec


def encode_zip_to_fast_acoustic_ofdm(
    zip_path: str,
    output_wav_path: str = "acoustic_archive.wav",
    sample_rate: int = 44100,
    n_fft: int = 1024,
    cp_len: int = 128,  # Echo guard cushion
    ecc_bytes: int = 32,
    block_size: int = 223,
):
    # 1. Read ZIP Binary
    with open(zip_path, "rb") as f:
        raw_bytes = f.read()

    print(f"Original ZIP Size: {len(raw_bytes)} bytes ({len(raw_bytes)/1e6:.2f} MB)")

    # 2. Add Header & Pad to Exact RS Block Boundary
    header = struct.pack(">I", len(raw_bytes))
    payload_with_header = header + raw_bytes

    pad_len_rs = (block_size - (len(payload_with_header) % block_size)) % block_size
    payload_padded = payload_with_header + b"\x00" * pad_len_rs

    # 3. Reed-Solomon Encoding
    print("Adding Reed-Solomon parity blocks...")
    rs = RSCodec(ecc_bytes)
    protected_payload = bytearray()
    for i in range(0, len(payload_padded), block_size):
        chunk = payload_padded[i : i + block_size]
        protected_payload.extend(rs.encode(chunk))

    print(f"Protected Payload Size: {len(protected_payload)} bytes")

    # 4. Convert Bytes to 16-QAM Nibbles
    byte_array = np.frombuffer(protected_payload, dtype=np.uint8)
    nibble_high = (byte_array >> 4) & 0x0F
    nibble_low = byte_array & 0x0F
    symbols = np.column_stack((nibble_high, nibble_low)).flatten()

    # 5. Define Extended Frequency Band (1.2 kHz to 13.8 kHz)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    active_carriers = np.where((freq_bins >= 1200) & (freq_bins <= 13800))[0]
    num_active = len(active_carriers)

    # Assign Pilot Carriers every 8th bin
    pilot_indices_in_active = np.arange(0, num_active, 8)
    data_indices_in_active = np.delete(np.arange(num_active), pilot_indices_in_active)
    num_data_carriers = len(data_indices_in_active)

    # 6. Normalized 16-QAM Constellation Table
    qam16_table = np.array(
        [
            -3 + 3j, -3 + 1j, -3 - 1j, -3 - 3j,
            -1 + 3j, -1 + 1j, -1 - 1j, -1 - 3j,
             1 + 3j,  1 + 1j,  1 - 1j,  1 - 3j,
             3 + 3j,  3 + 1j,  3 - 1j,  3 - 3j,
        ],
        dtype=np.complex64,
    ) / np.sqrt(10.0)

    total_ofdm_frames = int(np.ceil(len(symbols) / num_data_carriers))
    pad_symbols_len = (total_ofdm_frames * num_data_carriers) - len(symbols)
    symbols_padded = np.pad(symbols, (0, pad_symbols_len), mode="constant")

    qam_data = qam16_table[symbols_padded].reshape((total_ofdm_frames, num_data_carriers))

    # 7. Construct Frame Matrix with Embedded Comb Pilots
    active_matrix = np.zeros((total_ofdm_frames, num_active), dtype=np.complex64)
    active_matrix[:, pilot_indices_in_active] = 1.0 + 0.0j
    active_matrix[:, data_indices_in_active] = qam_data

    # 8. IFFT + Cyclic Prefix Generation
    spectrum = np.zeros((total_ofdm_frames, n_fft // 2 + 1), dtype=np.complex64)
    spectrum[:, active_carriers] = active_matrix
    time_frames = np.fft.irfft(spectrum, n=n_fft, axis=1)

    cp = time_frames[:, -cp_len:]
    frames_with_cp = np.hstack([cp, time_frames])
    payload_signal = frames_with_cp.flatten()

    # 9. Generate Distinctive Start and End Tones
    t_beep = np.arange(int(sample_rate * 0.2)) / float(sample_rate)
    start_beep = 0.5 * (np.sin(2 * np.pi * 7000 * t_beep) + np.sin(2 * np.pi * 10000 * t_beep))

    t_chirp = np.arange(int(sample_rate * 0.8)) / float(sample_rate)
    start_chirp = np.sin(2 * np.pi * (1200.0 + (13800.0 - 1200.0) * t_chirp / 2.0) * t_chirp)
    start_chirp = start_chirp * np.hanning(len(start_chirp))

    start_beacon = np.concatenate([start_beep, np.zeros(1000), start_chirp])

    # End Tone: Reverse Chirp + Double High Beep
    end_chirp = np.sin(2 * np.pi * (13800.0 + (1200.0 - 13800.0) * t_chirp / 2.0) * t_chirp)
    end_chirp = end_chirp * np.hanning(len(end_chirp))

    end_beacon = np.concatenate([end_chirp, np.zeros(1000), start_beep, np.zeros(1000), start_beep])

    silence = np.zeros(int(sample_rate * 0.3))
    full_signal = np.concatenate([silence, start_beacon, payload_signal, end_beacon, silence])

    # 10. Normalize and Write WAV
    max_val = np.max(np.abs(full_signal))
    scaled_signal = (full_signal / max_val) * 26214.0
    pcm_data = scaled_signal.astype(np.int16)

    with wave.open(output_wav_path, "w") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm_data.tobytes())

    duration = len(pcm_data) / sample_rate
    print(f"\nGenerated '{output_wav_path}' -> Total Duration: {duration / 60:.2f} minutes")


if __name__ == "__main__":
    encode_zip_to_fast_acoustic_ofdm("myfile.zip", "acoustic_archive.wav")