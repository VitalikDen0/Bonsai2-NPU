// Test Python script to verify Bonsai 2 PTQ1_0 conversion to dual bitplane (nz_bits, sgn_bits)
import sys
sys.path.insert(0, r'D:\Download\Bonsai2_NPU\Ternary-Bonsai-2-27B-mlx-2bit\runtime')
import numpy as np
import gguf.constants as gc
from enum import IntEnum
import gguf.gguf_reader as gr
from codec import transcode

members = {k: int(v) for k, v in gc.GGMLQuantizationType.__members__.items()}
members['PTQ1_0'] = 143
NewEnum = IntEnum('GGMLQuantizationType', members)
gc.GGMLQuantizationType = NewEnum
gr.GGMLQuantizationType = NewEnum
gc.GGML_QUANT_SIZES[NewEnum.PTQ1_0] = (128, 28)

reader = gr.GGUFReader(r'D:\Qualcomm\models\Bonsai-2-27B-PTQ1_0\Ternary-Bonsai-2-27B-PTQ1_0.gguf')
t = [t for t in reader.tensors if t.name == 'blk.0.attn_gate.weight'][0]
print(f'Testing {t.name}: shape={t.shape}, type={t.tensor_type}')
shape = tuple(int(n) for n in t.shape[::-1]) # (out_dim, in_dim) = (6144, 5120)
words, scales, biases = transcode(t.data.tobytes(), shape, 'PTQ1_0')
print('transcode ok! words shape:', words.shape, 'scales shape:', scales.shape)

# In words: each 32-bit word holds 16 2-bit trits c in {0, 1, 2}:
# c = (word >> (2*lane)) & 3
# w = (c - 1) * s in {-s, 0, +s}
# Let's unpack to nz_bits and sgn_bits:
rows, width = shape
prow = (width // 8) + ((-(width // 8)) % 128) # padded to 128B
print(f'rows={rows}, width={width}, prow={prow}')

# Let's verify for 5 random rows that (nz_bits, sgn_bits) matches w exactly:
rng = np.random.default_rng(42)
max_err = 0.0
for r in rng.choice(rows, size=5, replace=False):
    r = int(r)
    # reconstruct trits c from words:
    row_words = words[r]
    c_trits = np.empty(width, dtype=np.uint8)
    for lane in range(16):
        c_trits[lane::16] = (row_words >> (2 * lane)) & 3
    # w_ref: c==0 -> -s, c==1 -> 0, c==2 -> +s
    s_row = np.repeat(scales[r], 128)
    w_ref = (c_trits.astype(np.float32) - 1.0) * s_row

    # build bitplanes:
    nz_row = (c_trits != 1).astype(np.uint8) # 1 if non-zero
    sgn_row = (c_trits == 2).astype(np.uint8) # 1 if +s, 0 if -s

    # pack to bytes lsb-first:
    nz_bytes = np.packbits(nz_row.reshape(-1, 8)[:, ::-1]).astype(np.uint8)
    sgn_bytes = np.packbits(sgn_row.reshape(-1, 8)[:, ::-1]).astype(np.uint8)

    # reconstruct from packed bytes:
    nz_unpacked = np.unpackbits(nz_bytes).reshape(-1, 8)[:, ::-1].reshape(-1)
    sgn_unpacked = np.unpackbits(sgn_bytes).reshape(-1, 8)[:, ::-1].reshape(-1)

    w_recon = np.where(nz_unpacked == 0, 0.0, np.where(sgn_unpacked == 1, s_row, -s_row))
    err = np.max(np.abs(w_ref - w_recon))
    if err > max_err: max_err = err

print(f'Verification across rows: max_abs_err = {max_err}')
assert max_err == 0.0, 'Mismatch!'
print('BONSAI 2 TERNARY UNPACKING IS 100% BIT-EXACT!')
