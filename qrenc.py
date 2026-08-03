import struct
import numpy as np
import cv2
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H

EC_LEVELS = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}
HEADER_STRUCT = ">IIH"  # chunk_index (u32), total_chunks (u32), chunk_len (u16)
HEADER_LEN = struct.calcsize(HEADER_STRUCT)


def _build_qr_frame(payload: bytes, ec_level: str, box_size: int, border: int, frame_size: int):
    qr = qrcode.QRCode(
        error_correction=EC_LEVELS[ec_level],
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload, optimize=0)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    arr = np.array(img)
    # resize with nearest-neighbor so QR module edges stay crisp (no blur
    # from interpolation, which would hurt camera-decode reliability)
    resized = cv2.resize(arr, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def encode_file_to_qr_video(
    input_path: str,
    output_video_path: str = "qr_transfer.mp4",
    chunk_bytes: int = 1800,
    error_correction: str = "M",
    fps: int = 30,
    hold_seconds: float = 0.2,
    box_size: int = 6,
    border: int = 4,
    frame_size: int = 900,
):

    if error_correction not in EC_LEVELS:
        raise ValueError(f"error_correction must be one of {list(EC_LEVELS)}")

    with open(input_path, "rb") as f:
        raw = f.read()
    print(f"Input file size: {len(raw):,} bytes ({len(raw)/1e6:.2f} MB)")

    chunks = [raw[i:i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)] or [b""]
    total_chunks = len(chunks)
    print(f"Splitting into {total_chunks:,} QR frame(s) of up to {chunk_bytes} bytes each.")

    hold_frames = max(1, round(hold_seconds * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_size, frame_size))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open '{output_video_path}' for writing with OpenCV's mp4v codec. "
            f"Try a different output filename/extension, or update opencv-python "
            f"(pip install -U opencv-python)."
        )

    for idx, chunk in enumerate(chunks):
        header = struct.pack(HEADER_STRUCT, idx, total_chunks, len(chunk))
        payload = header + chunk
        frame = _build_qr_frame(payload, error_correction, box_size, border, frame_size)
        for _ in range(hold_frames):
            writer.write(frame)
        if (idx + 1) % 200 == 0 or idx == total_chunks - 1:
            print(f"  generated {idx+1:,}/{total_chunks:,} QR frames")

    writer.release()

    duration = total_chunks * hold_seconds
    throughput = len(raw) / duration if duration > 0 else 0
    print(f"\nGenerated '{output_video_path}' -> {duration/60:.2f} minutes of playback.")
    print(f"Effective throughput: ~{throughput/1024:.1f} KB/sec.")
    print("\nPlay this video FULL SCREEN on the sending machine. On the receiving "
          "machine, record the screen with a camera (steady mount, good lighting, "
          "no glare) for the whole duration, then decode the recording with qrdec.py.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode a file into a QR-code video for air-gapped transfer.")
    parser.add_argument("input_file", help="Path to the file to encode.")
    parser.add_argument("output_video", nargs="?", default="qr_transfer.mp4",
                         help="Output video path (default: qr_transfer.mp4).")
    parser.add_argument("--chunk-bytes", type=int, default=1800,
                         help="Payload bytes per QR frame (default: 1800).")
    parser.add_argument("--ec", choices=list(EC_LEVELS), default="M",
                         help="QR error correction level (default: M).")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30).")
    parser.add_argument("--hold-seconds", type=float, default=0.2,
                         help="How long each QR frame is displayed (default: 0.2).")
    parser.add_argument("--box-size", type=int, default=6, help="QR module pixel size (default: 6).")
    parser.add_argument("--border", type=int, default=4, help="QR quiet-zone border in modules (default: 4).")
    parser.add_argument("--frame-size", type=int, default=900, help="Output video is NxN pixels (default: 900).")
    args = parser.parse_args()

    encode_file_to_qr_video(
        args.input_file,
        args.output_video,
        chunk_bytes=args.chunk_bytes,
        error_correction=args.ec,
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        box_size=args.box_size,
        border=args.border,
        frame_size=args.frame_size,
    )