"""
Robust acoustic OFDM decoder (v2), matching ftoa_v2.py.

Key robustness features vs the original decoder:
  - Auto mono/stereo handling (the original bug: a stereo recording was silently
    misread as one long mono stream, which alone was enough to break decoding).
  - Periodic training-frame based channel estimation + timing re-lock, instead
    of trusting a single fixed frame spacing for the whole recording (drift
    accumulated unchecked in the original design).
  - QPSK demod (bigger noise margin than 16-QAM).
  - De-interleaving before Reed-Solomon, so burst errors get spread out instead
    of concentrated.
  - A redundant, independently-protected header section (decoded via majority
    vote across 3 tries) so a single bad block near the start doesn't sink the
    whole file.
"""

import struct
import wave
import numpy as np
from reedsolo import RSCodec, ReedSolomonError

# Must match the encoder exactly.
SAMPLE_RATE = 44100
N_FFT = 1024
CP_LEN = 384
ECC_BYTES = 64
BLOCK_SIZE = 191
TRAINING_INTERVAL = 16
INTERLEAVE_DEPTH = 32
FREQ_LOW, FREQ_HIGH = 1200.0, 13800.0

QPSK_TABLE = np.array(
    [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64
) / np.sqrt(2.0)


def _load_mono(wav_path):
    """Read a WAV file and return a mono float32 signal, downmixing if needed."""
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
        print(f"Note: input WAV has {nch} channels -- downmixed to mono.")
    return audio, sr


def _fast_cross_correlation(signal, template):
    n_sig = len(signal)
    n_tpl = len(template)
    n_fft = 1 << int(np.ceil(np.log2(n_sig + n_tpl - 1)))
    S = np.fft.rfft(signal, n=n_fft)
    T = np.fft.rfft(template, n=n_fft)
    corr_full = np.fft.irfft(S * np.conj(T), n=n_fft)
    return corr_full[: n_sig - n_tpl + 1]


def _active_carriers(n_fft=N_FFT, sr=SAMPLE_RATE):
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return np.where((freq_bins >= FREQ_LOW) & (freq_bins <= FREQ_HIGH))[0]


def _training_symbols(num_active, seed=1234):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, 4, size=num_active)
    return QPSK_TABLE[idx]


def _training_time_waveform(training_syms, active_carriers, n_fft, cp_len):
    spectrum = np.zeros(n_fft // 2 + 1, dtype=np.complex64)
    spectrum[active_carriers] = training_syms
    time_frame = np.fft.irfft(spectrum, n=n_fft)
    cp = time_frame[-cp_len:]
    return np.concatenate([cp, time_frame]).astype(np.float32)


def _deinterleave(data: bytes, depth: int, block_total: int, num_blocks: int):
    """Inverse of the encoder's block interleaver."""
    out = bytearray(num_blocks * block_total)
    pos = 0
    for start in range(0, num_blocks, depth):
        g = min(depth, num_blocks - start)
        for byte_pos in range(block_total):
            for k in range(g):
                blk_idx = start + k
                out[blk_idx * block_total + byte_pos] = data[pos]
                pos += 1
    return bytes(out)


def _extract_frame_fractional(audio, exact_pos, frame_len, guard=32):
    """Extract `frame_len` samples starting at a possibly-fractional sample
    position, using a Fourier-domain fractional delay (ideal band-limited
    interpolation) instead of rounding to the nearest integer sample.

    This matters a lot here: at our top carrier frequency (~13.8kHz, close to
    Nyquist/2 @ 44100Hz), a single sample of position error already causes
    close to a 90-degree phase rotation -- more than one QPSK quadrant. Real
    recording hardware will essentially never have its clock exactly locked
    to the assumed sample rate, so *some* fractional-sample drift is normal,
    and rounding to the nearest integer sample is not precise enough to
    equalize it out.
    """
    base = int(np.floor(exact_pos))
    frac = exact_pos - base
    lo = base - guard
    hi = base + frame_len + guard
    if lo < 0 or hi > len(audio):
        # fall back to plain rounding near the edges of the recording
        p = int(round(exact_pos))
        return audio[p:p + frame_len]
    seg = audio[lo:hi]
    n = len(seg)
    freqs = np.fft.fftfreq(n)
    spec = np.fft.fft(seg)
    shifted = np.real(np.fft.ifft(spec * np.exp(2j * np.pi * freqs * frac)))
    return shifted[guard:guard + frame_len]


def _qpsk_slice(symbols):
    dists = np.abs(symbols[:, np.newaxis] - QPSK_TABLE[np.newaxis, :])
    return np.argmin(dists, axis=1)


def decode_robust_acoustic_ofdm_to_zip(
    wav_path: str,
    output_zip_path: str = "recovered_myfile_v2.zip",
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    cp_len: int = CP_LEN,
    ecc_bytes: int = ECC_BYTES,
    block_size: int = BLOCK_SIZE,
    sync_search_radius: int = 40,
):
    print(f"Reading '{wav_path}'...")
    audio, sr = _load_mono(wav_path)
    print(f"Audio loaded: {len(audio):,} samples ({len(audio)/sr/60:.2f} minutes).")

    # --- locate start / end beacons (same scheme as before) -----------------
    t_chirp = np.arange(int(sr * 0.8)) / float(sr)
    ref_chirp = np.sin(2 * np.pi * (FREQ_LOW + (FREQ_HIGH - FREQ_LOW) * t_chirp / 2.0) * t_chirp)
    ref_chirp = (ref_chirp * np.hanning(len(ref_chirp))).astype(np.float32)
    corr_start = _fast_cross_correlation(audio, ref_chirp)
    sync_idx = int(np.argmax(corr_start))
    payload_start = sync_idx + len(ref_chirp)
    print(f"-> Start beacon locked at sample {sync_idx} ({sync_idx/sr:.2f}s).")

    ref_end_chirp = np.sin(2 * np.pi * (FREQ_HIGH + (FREQ_LOW - FREQ_HIGH) * t_chirp / 2.0) * t_chirp)
    ref_end_chirp = (ref_end_chirp * np.hanning(len(ref_end_chirp))).astype(np.float32)
    corr_end = _fast_cross_correlation(audio[payload_start:], ref_end_chirp)
    payload_length = int(np.argmax(corr_end))
    print(f"-> End beacon locked; payload region: {payload_length:,} samples.")

    # --- carrier / training setup --------------------------------------------
    active_carriers = _active_carriers(n_fft, sr)
    num_active = len(active_carriers)
    training_syms = _training_symbols(num_active)
    training_waveform = _training_time_waveform(training_syms, active_carriers, n_fft, cp_len)

    frame_len = n_fft + cp_len
    total_frames = payload_length // frame_len
    is_training = [(i % TRAINING_INTERVAL == 0) for i in range(total_frames)]
    training_idx = [i for i, t in enumerate(is_training) if t]
    data_idx = [i for i, t in enumerate(is_training) if not t]
    print(f"Processing {total_frames:,} OFDM frames "
          f"({len(training_idx)} training / {len(data_idx)} data)...")

    # --- fine timing recovery at every training frame -------------------------
    # Nominal (drift-free) position, then locally search +/- sync_search_radius
    # samples for the best-correlated position against the known clean training
    # waveform. This re-locks timing periodically instead of trusting one fixed
    # spacing for the whole file.
    actual_pos = {}
    running_ref_pos = payload_start   # position of the *previous* training frame (or payload start)
    running_ref_idx = 0
    for i in training_idx:
        # Predict this training frame's position from the last *found* one, not
        # from the fixed original reference -- only the drift accumulated since
        # the last checkpoint needs to fit inside the search radius, not the
        # drift accumulated over the whole recording.
        nominal = running_ref_pos + (i - running_ref_idx) * frame_len
        lo = max(0, nominal - sync_search_radius)
        hi = min(len(audio) - len(training_waveform), nominal + sync_search_radius)
        if hi <= lo:
            actual_pos[i] = nominal
        else:
            window = audio[lo:hi + len(training_waveform)]
            corr = _fast_cross_correlation(window, training_waveform)
            best = lo + int(np.argmax(corr))
            actual_pos[i] = best
        running_ref_pos = actual_pos[i]
        running_ref_idx = i

    if not training_idx:
        raise ValueError("No training frames found -- payload region looks too short.")

    # Interpolate frame positions for every frame (training frames pinned to
    # their re-locked position, data frames linearly interpolated between the
    # two bracketing training frames; edges extrapolated from nearest slope).
    frame_pos = np.zeros(total_frames)
    t_arr = np.array(training_idx)
    p_arr = np.array([actual_pos[i] for i in training_idx])
    for i in range(total_frames):
        if i <= t_arr[0]:
            # extrapolate before first training frame using first two points
            if len(t_arr) >= 2:
                slope = (p_arr[1] - p_arr[0]) / (t_arr[1] - t_arr[0])
            else:
                slope = frame_len
            frame_pos[i] = p_arr[0] + slope * (i - t_arr[0])
        elif i >= t_arr[-1]:
            if len(t_arr) >= 2:
                slope = (p_arr[-1] - p_arr[-2]) / (t_arr[-1] - t_arr[-2])
            else:
                slope = frame_len
            frame_pos[i] = p_arr[-1] + slope * (i - t_arr[-1])
        else:
            j = np.searchsorted(t_arr, i)
            t0, t1 = t_arr[j - 1], t_arr[j]
            p0, p1 = p_arr[j - 1], p_arr[j]
            frame_pos[i] = p0 + (p1 - p0) * (i - t0) / (t1 - t0)
    frame_pos_exact = frame_pos  # keep float precision; see _extract_frame_fractional

    # --- per-frame FFT + channel estimate at training frames -------------------
    active_range = np.arange(num_active)
    H_train = {}
    for i in training_idx:
        pos = frame_pos_exact[i]
        if pos < 0 or pos + frame_len > len(audio):
            continue
        frame = _extract_frame_fractional(audio, pos, frame_len)
        spectrum = np.fft.rfft(frame[cp_len:])
        rx_active = spectrum[active_carriers]
        H_train[i] = rx_active / training_syms

    if not H_train:
        raise ValueError("Could not extract any valid training frames from the recording.")

    # Smooth the per-checkpoint channel estimates over a small window of
    # neighboring training frames. A single training frame's estimate carries
    # its own measurement noise; averaging several nearby ones (the channel
    # itself changes slowly for a stationary acoustic setup) suppresses that
    # noise instead of injecting it into every data frame between checkpoints.
    SMOOTH_WINDOW = 5
    keys_sorted = sorted(H_train.keys())
    H_smoothed = {}
    for idx_pos, k in enumerate(keys_sorted):
        lo = max(0, idx_pos - SMOOTH_WINDOW // 2)
        hi = min(len(keys_sorted), idx_pos + SMOOTH_WINDOW // 2 + 1)
        neighborhood = [H_train[keys_sorted[j]] for j in range(lo, hi)]
        H_smoothed[k] = np.mean(neighborhood, axis=0)
    H_train = H_smoothed

    def channel_for_frame(i):
        """Time-interpolated channel estimate for data frame i, from the
        bracketing training-frame estimates."""
        keys = sorted(H_train.keys())
        if i <= keys[0]:
            return H_train[keys[0]]
        if i >= keys[-1]:
            return H_train[keys[-1]]
        j = np.searchsorted(keys, i)
        t0, t1 = keys[j - 1], keys[j]
        w = (i - t0) / (t1 - t0)
        return H_train[t0] * (1 - w) + H_train[t1] * w

    # --- demodulate data frames --------------------------------------------
    all_symbol_idx = []
    for i in data_idx:
        pos = frame_pos_exact[i]
        if pos < 0 or pos + frame_len > len(audio):
            continue
        frame = _extract_frame_fractional(audio, pos, frame_len)
        spectrum = np.fft.rfft(frame[cp_len:])
        rx_active = spectrum[active_carriers]
        H = channel_for_frame(i)
        eq = rx_active / H
        all_symbol_idx.append(_qpsk_slice(eq))

    symbol_idx = np.concatenate(all_symbol_idx) if all_symbol_idx else np.array([], dtype=int)
    print(f"Demodulated {len(symbol_idx):,} QPSK symbols.")

    # Symbols -> bytes (4 symbols/byte, 2 bits each)
    n_full_bytes = (len(symbol_idx) // 4) * 4
    symbol_idx = symbol_idx[:n_full_bytes]
    s = symbol_idx.reshape(-1, 4)
    rx_bytes = bytes(((s[:, 0] << 6) | (s[:, 1] << 4) | (s[:, 2] << 2) | s[:, 3]).astype(np.uint8))

    block_total = block_size + ecc_bytes
    rs = RSCodec(ecc_bytes)

    # --- header section: 3 independent copies, NOT interleaved ---------------
    header_section_len = 3 * block_total
    header_bytes = rx_bytes[:header_section_len]
    orig_len_candidates = []
    header_ok = 0
    for k in range(3):
        chunk = header_bytes[k * block_total:(k + 1) * block_total]
        if len(chunk) < block_total:
            continue
        try:
            clean = rs.decode(chunk)
            clean_block = clean[0] if isinstance(clean, tuple) else clean
            orig_len_candidates.append(struct.unpack(">I", bytes(clean_block[:4]))[0])
            header_ok += 1
        except ReedSolomonError:
            continue

    if orig_len_candidates:
        # majority vote, ties broken by first candidate
        vals, counts = np.unique(orig_len_candidates, return_counts=True)
        orig_len = int(vals[np.argmax(counts)])
        print(f"Header decoded ({header_ok}/3 copies agreed): target size {orig_len:,} bytes.")
    else:
        orig_len = None
        print("WARNING: all 3 header copies failed Reed-Solomon -- file size unknown, "
              "will use best-effort recovered length.")

    # --- main payload: de-interleave then RS-decode block by block -----------
    main_bytes = rx_bytes[header_section_len:]
    observed_num_blocks = len(main_bytes) // block_total
    if orig_len is not None:
        # Authoritative block count from the (reliably-decoded) header length,
        # rather than the observed frame/byte count. A drifted/resampled
        # recording can genuinely have one fewer usable frame at the tail than
        # the nominal design assumed; trusting the observed count then
        # misaligns the de-interleaving for the *entire* last group, not just
        # the missing bytes.
        num_main_blocks = -(-orig_len // block_size)  # ceil division
        needed_bytes = num_main_blocks * block_total
        if len(main_bytes) < needed_bytes:
            main_bytes = main_bytes + b"\x00" * (needed_bytes - len(main_bytes))
        else:
            main_bytes = main_bytes[:needed_bytes]
    else:
        num_main_blocks = observed_num_blocks
        main_bytes = main_bytes[:num_main_blocks * block_total]

    def _rs_decode_blocks(nb, mbytes):
        deint = _deinterleave(mbytes, INTERLEAVE_DEPTH, block_total, nb)
        out = bytearray()
        ok_c, fail_c = 0, 0
        for i in range(nb):
            chunk = deint[i * block_total:(i + 1) * block_total]
            try:
                clean = rs.decode(chunk)
                clean_block = clean[0] if isinstance(clean, tuple) else clean
                out.extend(clean_block)
                ok_c += 1
            except ReedSolomonError:
                out.extend(b"\x00" * block_size)
                fail_c += 1
        return out, ok_c, fail_c

    if orig_len is not None:
        decoded, ok, failed = _rs_decode_blocks(num_main_blocks, main_bytes)
    else:
        # No reliable header: blindly trusting the observed block count risks
        # misaligning the whole trailing interleave group if drift shifted the
        # frame count by one. Try a small neighborhood of block counts and
        # keep whichever yields the most successful RS blocks.
        raw_main_bytes = rx_bytes[header_section_len:]
        best = None
        for delta in (0, -1, 1, -2, 2):
            candidate_nb = observed_num_blocks + delta
            if candidate_nb <= 0:
                continue
            needed = candidate_nb * block_total
            padded = raw_main_bytes[:needed]
            if len(padded) < needed:
                padded = padded + b"\x00" * (needed - len(padded))
            out, ok_c, fail_c = _rs_decode_blocks(candidate_nb, padded)
            if best is None or ok_c > best[1]:
                best = (out, ok_c, fail_c, candidate_nb)
        decoded, ok, failed, num_main_blocks = best

    print(f"RS summary (main payload): {ok:,} blocks OK, {failed:,} blocks failed "
          f"out of {num_main_blocks:,} ({100*ok/max(1,num_main_blocks):.1f}% OK).")

    if orig_len is not None:
        zip_bytes = bytes(decoded[:orig_len])
    else:
        # best effort: trim trailing zero padding
        zip_bytes = bytes(decoded).rstrip(b"\x00")

    with open(output_zip_path, "wb") as f:
        f.write(zip_bytes)

    print(f"\nWrote '{output_zip_path}' ({len(zip_bytes):,} bytes).")
    return dict(ok=ok, failed=failed, total=num_main_blocks, orig_len=orig_len,
                header_ok=header_ok)


if __name__ == "__main__":
    decode_robust_acoustic_ofdm_to_zip(
        wav_path="acoustic_archive_v2.wav",
        output_zip_path="recovered_myfile_v2.zip",
    )
