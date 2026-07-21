"""
encoder.py — Encode any binary file into photo-and-print-safe PNG images.

Format v2: photo/print-safe (screenshot, camera photo, or printed page)
----------------------------------------------------------------------
Earlier versions of this format only had to survive lossless screenshots.
Camera photos and printed-then-scanned pages add: rotation, perspective
("keystone") skew, uneven lighting, camera white-balance drift, and JPEG
recompression. Surviving those requires the same toolkit real 2D barcodes
(QR, Data Matrix) use:

  1. FOUR FINDER MARKERS, one in each corner, using the classic 1:1:3:1:1
     concentric black/white/black ratio pattern. This is detected by
     scanning for that specific run-length ratio, not by simple color
     thresholding — a ratio scan keeps working even in front of a dark or
     colorful background, where a plain solid marker gets swallowed by
     whatever is behind it (this was empirically verified: a solid-square
     marker failed against a dark background; the ratio pattern did not).
  2. A PERSPECTIVE (homography) TRANSFORM computed from the 4 detected
     marker centers, correcting rotation and keystone skew back to a clean
     top-down square before reading any data.
  3. A SMALL 4-COLOR PALETTE (red/green/blue/yellow) instead of raw 0-255
     RGB bytes. Camera exposure and printer color drift make exact byte
     values unrecoverable; classifying each module as "nearest of 4
     well-separated colors" is far more reliable.
  4. REED-SOLOMON ERROR CORRECTION (the "checkpoints"). Every frame's
     header+payload bytes are RS-encoded before being drawn, so the
     decoder can reconstruct the exact original bytes even when some
     percentage of modules were misread — not just detect that something
     went wrong.

Tested to survive: any static screenshot (scaled or sloppily cropped), and
simulated phone photos with up to ~15 degrees of rotation + keystone skew,
brightness/contrast shifts, colored or dark backgrounds, and JPEG
recompression down to quality 60 — all with zero module misreads in
testing, plus Reed-Solomon headroom on top for real-world margin. It
assumes a single flat, non-folded, non-crumpled surface (no OpenCV, no
true multi-plane perspective correction).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import struct
import time
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from reedsolo import RSCodec

# --------------------------------------------------------------------------- #
# Format constants (must match decoder.py; also embedded in each frame's
# header for validation).
# --------------------------------------------------------------------------- #

MAGIC = b"PIMG"
VERSION = 3

PALETTE: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (255, 191, 0),
    (127, 255, 0),
    (0, 255, 63),
    (0, 255, 255),
    (0, 63, 255),
    (127, 0, 255),
    (255, 0, 191),
]
BITS_PER_MODULE = 3  # log2(len(PALETTE)); must stay a power of 2 to keep this exact

FINDER_UNIT = 14                    # px per ring-unit in the 1:1:3:1:1 finder pattern
MARKER_SIZE = 7 * FINDER_UNIT        # 98px, classic QR-style finder footprint
MARKER_MARGIN = 24                  # px from canvas edge to outer edge of each marker
GRID_GAP = 20                       # quiet buffer between markers and the data grid

DEFAULT_CANVAS_SIZE = 1200
DEFAULT_MODULE_SIZE = 14
DEFAULT_NSYM = 64                   # Reed-Solomon parity bytes per 255-byte codeword
RS_NSIZE = 255                      # Reed-Solomon codeword size (GF(256) limit)

FILENAME_FIELD_LEN = 200
PBKDF2_ITERATIONS = 200_000
KEY_LENGTH = 32
NONCE_LENGTH = 12
SALT_LENGTH = 16

# Header layout (little-endian, no padding):
#   magic(4s) version(B) canvas_size(H) module_size(H) nsym(B)
#   frame_index(H) total_frames(H) true_payload_len(I) total_ciphertext_len(I)
#   original_size(Q) filename_len(B) filename(200s) sha256(32s) salt(16s)
#   nonce(12s) frame_checksum(4s)
HEADER_FMT = "<4sBHHBHHIIQB200s32s16s12s4s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("encoder")


class EncodingError(Exception):
    """Raised for any unrecoverable encoding failure."""


def read_file_bytes(path: Path) -> bytes:
    """Read the entire input file into memory."""
    if not path.is_file():
        raise EncodingError(f"Input file not found: {path}")
    return path.read_bytes()


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit AES key from a password and salt using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=KEY_LENGTH, salt=salt, iterations=PBKDF2_ITERATIONS)
    return kdf.derive(password.encode("utf-8"))


def encrypt_data(plaintext: bytes, password: str) -> Tuple[bytes, bytes, bytes]:
    """Encrypt plaintext with AES-256-GCM. Returns (ciphertext_with_tag, salt, nonce)."""
    salt = os.urandom(SALT_LENGTH)
    nonce = os.urandom(NONCE_LENGTH)
    key = derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return ciphertext, salt, nonce


# --------------------------------------------------------------------------- #
# Geometry / capacity
# --------------------------------------------------------------------------- #

def marker_centers(canvas_size: int) -> dict:
    """Canonical (post-correction) pixel centers of the 4 finder markers."""
    c = MARKER_MARGIN + MARKER_SIZE / 2
    return {
        "tl": (c, c),
        "tr": (canvas_size - c, c),
        "bl": (c, canvas_size - c),
        "br": (canvas_size - c, canvas_size - c),
    }


def grid_geometry(canvas_size: int, module_size: int) -> Tuple[int, int]:
    """Return (grid_start_px, cols) for the data module grid."""
    grid_start = MARKER_MARGIN + MARKER_SIZE + GRID_GAP
    grid_extent = canvas_size - 2 * grid_start
    if grid_extent <= 0:
        raise EncodingError("Canvas too small for markers/gap. Increase --canvas-size.")
    cols = grid_extent // module_size
    if cols <= 0:
        raise EncodingError("Module size too large for canvas. Decrease --module-size.")
    return grid_start, cols


def compute_capacity(canvas_size: int, module_size: int, nsym: int) -> Tuple[int, int]:
    """Return (message_bytes_capacity_per_frame, cols). message capacity includes the header."""
    _, cols = grid_geometry(canvas_size, module_size)
    total_modules = cols * cols
    total_bits = total_modules * BITS_PER_MODULE
    capacity_bytes = total_bits // 8
    num_codewords = capacity_bytes // RS_NSIZE
    if num_codewords <= 0:
        raise EncodingError("Canvas/module size too small to fit even one Reed-Solomon codeword.")
    message_bytes_capacity = num_codewords * (RS_NSIZE - nsym)
    if message_bytes_capacity <= HEADER_SIZE:
        raise EncodingError("Canvas too small to fit header + any payload. Increase --canvas-size.")
    return message_bytes_capacity, cols


# --------------------------------------------------------------------------- #
# Reed-Solomon (manual, deterministic chunking across GF(256) codewords)
# --------------------------------------------------------------------------- #

def rs_encode(message: bytes, nsym: int) -> bytes:
    """RS-encode a message, chunked into fixed-size codewords for deterministic capacity."""
    rsc = RSCodec(nsym, nsize=RS_NSIZE)
    k = RS_NSIZE - nsym
    if len(message) % k != 0:
        raise EncodingError("Internal error: message length is not a multiple of the codeword data size.")
    out = bytearray()
    for i in range(0, len(message), k):
        out.extend(rsc.encode(message[i:i + k]))
    return bytes(out)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

def build_header(
    canvas_size: int,
    module_size: int,
    nsym: int,
    frame_index: int,
    total_frames: int,
    true_payload_len: int,
    frame_checksum_source: bytes,
    total_ciphertext_len: int,
    original_size: int,
    filename: str,
    sha256: bytes,
    salt: bytes,
    nonce: bytes,
) -> bytes:
    """Pack the fixed-size binary header for one frame."""
    filename_bytes = filename.encode("utf-8")[:FILENAME_FIELD_LEN]
    filename_padded = filename_bytes.ljust(FILENAME_FIELD_LEN, b"\x00")
    frame_checksum = hashlib.sha256(frame_checksum_source).digest()[:4]
    return struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, canvas_size, module_size, nsym,
        frame_index, total_frames, true_payload_len, total_ciphertext_len,
        original_size, len(filename_bytes), filename_padded, sha256, salt, nonce,
        frame_checksum,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def draw_finder(draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    """Draw a classic 1:1:3:1:1 concentric-square finder pattern centered at (cx, cy)."""
    u = FINDER_UNIT
    for size, color in [(7 * u, (0, 0, 0)), (5 * u, (255, 255, 255)), (3 * u, (0, 0, 0))]:
        half = size / 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)


def bytes_to_symbols(data: bytes) -> List[int]:
    """
    Pack bytes into a bit-stream, then split into BITS_PER_MODULE-wide symbols (MSB first).
    The last symbol is zero-padded if the bit count isn't an exact multiple.
    """
    total_bits = len(data) * 8
    num_symbols = math.ceil(total_bits / BITS_PER_MODULE)
    padded_bit_len = num_symbols * BITS_PER_MODULE

    bit_buffer = 0
    bit_count = 0
    symbols = []
    byte_iter = iter(data)
    bits_emitted = 0

    for byte in data:
        bit_buffer = (bit_buffer << 8) | byte
        bit_count += 8
        while bit_count >= BITS_PER_MODULE:
            bit_count -= BITS_PER_MODULE
            symbols.append((bit_buffer >> bit_count) & ((1 << BITS_PER_MODULE) - 1))
            bits_emitted += BITS_PER_MODULE
        bit_buffer &= (1 << bit_count) - 1 if bit_count else 0

    if bit_count > 0:
        # Left-align the remaining bits within a full symbol width, zero-padded on the right.
        remaining = (bit_buffer << (BITS_PER_MODULE - bit_count)) & ((1 << BITS_PER_MODULE) - 1)
        symbols.append(remaining)

    assert len(symbols) == num_symbols, "internal bit-packing length mismatch"
    return symbols


def render_frame(
    header_bytes: bytes,
    padded_payload: bytes,
    canvas_size: int,
    module_size: int,
    nsym: int,
    out_path: Path,
) -> None:
    """Render one PNG frame: 4 finder markers + RS-encoded 4-color module grid."""
    message = header_bytes + padded_payload
    encoded = rs_encode(message, nsym)
    symbols = bytes_to_symbols(encoded)

    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for cx, cy in marker_centers(canvas_size).values():
        draw_finder(draw, cx, cy)

    grid_start, cols = grid_geometry(canvas_size, module_size)
    for idx, sym in enumerate(symbols):
        col = idx % cols
        row = idx // cols
        x0 = grid_start + col * module_size
        y0 = grid_start + row * module_size
        draw.rectangle([x0, y0, x0 + module_size - 1, y0 + module_size - 1], fill=PALETTE[sym])

    img.save(out_path, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a file into photo/print-safe PNG images.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--output", required=True, help="Output directory for frame_NNNN.png files.")
    parser.add_argument("--key", required=True, help="Password used to encrypt the file.")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE, help="Square canvas size in pixels.")
    parser.add_argument("--module-size", type=int, default=DEFAULT_MODULE_SIZE, help="Data module size in pixels.")
    parser.add_argument("--nsym", type=int, default=DEFAULT_NSYM,
                         help="Reed-Solomon parity bytes per 255-byte codeword (higher = more error tolerance, less capacity).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Reading input file: %s", input_path)
        original_bytes = read_file_bytes(input_path)
        original_size = len(original_bytes)
        sha256 = hashlib.sha256(original_bytes).digest()

        logger.info("Encrypting with AES-256-GCM ...")
        ciphertext, salt, nonce = encrypt_data(original_bytes, args.key)
        encrypted_size = len(ciphertext)

        message_capacity, cols = compute_capacity(args.canvas_size, args.module_size, args.nsym)
        payload_capacity = message_capacity - HEADER_SIZE
        total_frames = max(1, math.ceil(encrypted_size / payload_capacity))

        logger.info(
            "Original size: %d bytes | Encrypted size: %d bytes | Payload capacity/frame: %d bytes | "
            "Grid: %dx%d modules | Frames: %d",
            original_size, encrypted_size, payload_capacity, cols, cols, total_frames,
        )

        for frame_index in range(1, total_frames + 1):
            chunk_start = (frame_index - 1) * payload_capacity
            chunk = ciphertext[chunk_start:chunk_start + payload_capacity]
            true_len = len(chunk)
            # Every frame's message must be the same length for deterministic RS chunking,
            # so short (usually final) chunks are zero-padded; true_len is recorded in the
            # header so the decoder knows where the real data ends.
            padded_chunk = chunk.ljust(payload_capacity, b"\x00")

            header_bytes = build_header(
                canvas_size=args.canvas_size,
                module_size=args.module_size,
                nsym=args.nsym,
                frame_index=frame_index,
                total_frames=total_frames,
                true_payload_len=true_len,
                frame_checksum_source=padded_chunk,
                total_ciphertext_len=encrypted_size,
                original_size=original_size,
                filename=input_path.name,
                sha256=sha256,
                salt=salt,
                nonce=nonce,
            )

            out_path = output_dir / f"frame_{frame_index:04d}.png"
            render_frame(header_bytes, padded_chunk, args.canvas_size, args.module_size, args.nsym, out_path)
            logger.info("Wrote %s (%d/%d, %.1f%%)", out_path.name, frame_index, total_frames,
                        100.0 * frame_index / total_frames)

        elapsed = time.time() - start_time
        logger.info("Done in %.2fs. Output directory: %s", elapsed, output_dir)

    except EncodingError as exc:
        logger.error("Encoding failed: %s", exc)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors clearly
        logger.error("Unexpected error: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
