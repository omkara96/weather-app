"""
Robust acoustic OFDM encoder (v2).

Changes vs the original ftoa.py, all aimed at surviving a real speaker -> room -> mic
path instead of only a clean digital channel:

  1. QPSK instead of 16-QAM
       - 2 bits/symbol instead of 4, but ~3x the noise margin (minimum distance
         between constellation points is much larger for the same average power).
       - This is the single biggest lever for surviving acoustic noise.

  2. Much longer cyclic prefix (guard interval)
       - 384 samples (~8.7ms @ 44100Hz) instead of 128 (~2.9ms).
       - A typical room's early reflections/reverb tail is on the order of
         10-50ms. The original guard interval was too short to contain that,
         which destroys the whole cyclic-prefix trick and causes inter-symbol
         interference no frequency-domain equalizer can undo.

  3. Periodic full-channel "training frames" instead of sparse comb pilots
       - Every TRAINING_INTERVAL-th frame carries a known pseudo-random QPSK
         pattern on *every* active subcarrier (not just every 8th bin).
       - The decoder gets an accurate, full-resolution channel snapshot at
         each training frame, and can also use it to re-lock frame timing
         periodically -- capping how much clock drift/jitter can accumulate
         between corrections (this was the actual root cause of the original
         decoder's failures on a real recording: timing drift grew unchecked
         over thousands of frames).
       - All non-training frames carry 100% data (no per-frame pilots wasted).

  4. Byte interleaving across many Reed-Solomon blocks
       - A noise spike or dropout in the recording corrupts a contiguous
         stretch of *symbols*. Without interleaving that lands as a burst of
         byte errors concentrated in one or two RS blocks, which can exceed
         what that block can correct even though the *average* error rate
         is fine.
       - Interleaving spreads consecutive transmitted bytes across many
         different RS blocks, turning a burst error into isolated
         single-byte errors in many blocks -- exactly what Reed-Solomon is
         good at fixing.

  5. Stronger Reed-Solomon (more parity bytes per block).

  6. The header (payload length) is repeated several times so a single
     lost block can't take down the whole decode.
"""

import struct
import wave
import numpy as np
from reedsolo import RSCodec

# ---------------------------------------------------------------------------
# Tunable parameters (must match on encoder + decoder)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
N_FFT = 1024
CP_LEN = 384                # guard interval, ~8.7ms @ 44100Hz
ECC_BYTES = 64              # RS parity bytes per block (was 32)
BLOCK_SIZE = 191            # RS data bytes per block (was 223) -> RS(255,191)
TRAINING_INTERVAL = 16      # every Nth frame is a full-channel training frame
INTERLEAVE_DEPTH = 32       # how many RS blocks get byte-interleaved together
FREQ_LOW, FREQ_HIGH = 1200.0, 13800.0

QPSK_TABLE = np.array(
    [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64
) / np.sqrt(2.0)


def _active_carriers(n_fft=N_FFT, sr=SAMPLE_RATE):
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return np.where((freq_bins >= FREQ_LOW) & (freq_bins <= FREQ_HIGH))[0]


def _training_symbols(num_active, seed=1234):
    """Deterministic pseudo-random QPSK training pattern, same on both ends."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, 4, size=num_active)
    return QPSK_TABLE[idx]


def _rs_encode_header_section(raw_len: int, ecc_bytes: int, block_size: int, rs: RSCodec):
    """Encode the payload length as 3 INDEPENDENT RS blocks (not interleaved with
    the main payload). Each copy is a full, separately-protected block, so the
    header survives unless all 3 copies happen to fail -- much stronger than
    just repeating the 4 bytes inline in one block."""
    header = struct.pack(">I", raw_len)
    header_padded = header + b"\x00" * (block_size - len(header))
    header_block_encoded = rs.encode(header_padded)
    return bytes(header_block_encoded) * 3


def _rs_encode_main_payload(raw_bytes: bytes, ecc_bytes: int, block_size: int, rs: RSCodec):
    pad_len = (block_size - (len(raw_bytes) % block_size)) % block_size
    payload_padded = raw_bytes + b"\x00" * pad_len

    protected = bytearray()
    for i in range(0, len(payload_padded), block_size):
        chunk = payload_padded[i : i + block_size]
        protected.extend(rs.encode(chunk))
    return bytes(protected)


def _interleave(data: bytes, depth: int, block_total: int):
    """Block-interleave RS codewords so a burst error spreads across many blocks.

    `data` is a sequence of RS-encoded blocks each `block_total` bytes long.
    We group blocks into chunks of `depth` blocks, and within each chunk we
    transpose: write out byte 0 of every block, then byte 1 of every block, etc.
    A contiguous burst corruption in the transmitted stream then hits at most
    one byte in each of many blocks, instead of many bytes in one block.
    """
    num_blocks = len(data) // block_total
    out = bytearray()
    for start in range(0, num_blocks, depth):
        group = [data[(start + k) * block_total : (start + k + 1) * block_total]
                  for k in range(min(depth, num_blocks - start))]
        g = len(group)
        for byte_pos in range(block_total):
            for blk in group:
                out.append(blk[byte_pos])
    return bytes(out)


def encode_zip_to_robust_acoustic_ofdm(
    zip_path: str,
    output_wav_path: str = "acoustic_archive_v2.wav",
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    cp_len: int = CP_LEN,
    ecc_bytes: int = ECC_BYTES,
    block_size: int = BLOCK_SIZE,
):
    with open(zip_path, "rb") as f:
        raw_bytes = f.read()
    print(f"Original ZIP Size: {len(raw_bytes)} bytes ({len(raw_bytes)/1e6:.2f} MB)")

    block_total = block_size + ecc_bytes
    rs = RSCodec(ecc_bytes)
    print("Adding Reed-Solomon parity (stronger: %d ecc / %d data bytes)..." % (ecc_bytes, block_size))
    header_section = _rs_encode_header_section(len(raw_bytes), ecc_bytes, block_size, rs)
    protected_payload = _rs_encode_main_payload(raw_bytes, ecc_bytes, block_size, rs)
    print(f"Protected Payload Size: {len(protected_payload)} bytes (+{len(header_section)} header bytes)")

    print(f"Interleaving across groups of {INTERLEAVE_DEPTH} RS blocks...")
    interleaved = header_section + _interleave(protected_payload, INTERLEAVE_DEPTH, block_total)

    # Bytes -> QPSK symbols (2 bits per symbol => 4 symbols per byte)
    byte_array = np.frombuffer(interleaved, dtype=np.uint8)
    b0 = (byte_array >> 6) & 0x03
    b1 = (byte_array >> 4) & 0x03
    b2 = (byte_array >> 2) & 0x03
    b3 = byte_array & 0x03
    symbol_idx = np.column_stack((b0, b1, b2, b3)).flatten()

    active_carriers = _active_carriers(n_fft, sample_rate)
    num_active = len(active_carriers)
    training_syms = _training_symbols(num_active)

    num_data_frames_needed = int(np.ceil(len(symbol_idx) / num_active))
    pad_syms = num_data_frames_needed * num_active - len(symbol_idx)
    symbol_idx = np.pad(symbol_idx, (0, pad_syms), mode="constant")
    data_qam = QPSK_TABLE[symbol_idx].reshape((num_data_frames_needed, num_active))

    # Interleave training frames among data frames: 1 training frame every
    # TRAINING_INTERVAL frames.
    frames = []
    data_i = 0
    frame_i = 0
    while data_i < num_data_frames_needed:
        if frame_i % TRAINING_INTERVAL == 0:
            frames.append(training_syms.copy())
        else:
            frames.append(data_qam[data_i])
            data_i += 1
        frame_i += 1
    active_matrix = np.array(frames, dtype=np.complex64)
    total_ofdm_frames = active_matrix.shape[0]
    print(f"Total OFDM frames: {total_ofdm_frames} "
          f"({num_data_frames_needed} data + {total_ofdm_frames - num_data_frames_needed} training)")

    # IFFT + cyclic prefix
    spectrum = np.zeros((total_ofdm_frames, n_fft // 2 + 1), dtype=np.complex64)
    spectrum[:, active_carriers] = active_matrix
    time_frames = np.fft.irfft(spectrum, n=n_fft, axis=1)
    cp = time_frames[:, -cp_len:]
    frames_with_cp = np.hstack([cp, time_frames])
    payload_signal = frames_with_cp.flatten()

    # Start/end beacons (unchanged design: distinctive beep + chirp)
    t_beep = np.arange(int(sample_rate * 0.2)) / float(sample_rate)
    start_beep = 0.5 * (np.sin(2 * np.pi * 7000 * t_beep) + np.sin(2 * np.pi * 10000 * t_beep))

    t_chirp = np.arange(int(sample_rate * 0.8)) / float(sample_rate)
    start_chirp = np.sin(2 * np.pi * (FREQ_LOW + (FREQ_HIGH - FREQ_LOW) * t_chirp / 2.0) * t_chirp)
    start_chirp = start_chirp * np.hanning(len(start_chirp))
    start_beacon = np.concatenate([start_beep, np.zeros(1000), start_chirp])

    end_chirp = np.sin(2 * np.pi * (FREQ_HIGH + (FREQ_LOW - FREQ_HIGH) * t_chirp / 2.0) * t_chirp)
    end_chirp = end_chirp * np.hanning(len(end_chirp))
    end_beacon = np.concatenate([end_chirp, np.zeros(1000), start_beep, np.zeros(1000), start_beep])

    silence = np.zeros(int(sample_rate * 0.3))
    full_signal = np.concatenate([silence, start_beacon, payload_signal, end_beacon, silence])

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
    encode_zip_to_robust_acoustic_ofdm("myfile.zip", "acoustic_archive_v2.wav")
