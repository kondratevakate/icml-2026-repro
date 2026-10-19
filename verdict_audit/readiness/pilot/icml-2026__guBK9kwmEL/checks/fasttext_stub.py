"""Offline stand-in for the 6.7 GB fastText model that tarte_ai.TARTE_TablePreprocessor loads
(hi-paris/fastText cc.en.300.bin).  It returns a deterministic pseudo-random 300-d vector per string
(seeded by CRC32 of the string).  The synthetic data has no semantics, so word-embedding semantics are irrelevant
here; everything downstream of the preprocessor (masking of missing attributes, TARTE, fusion, losses) is unchanged."""
import zlib
import numpy as np


class StubFastText:
    def get_sentence_vector(self, s):
        rng = np.random.default_rng(zlib.crc32(str(s).encode("utf-8")))
        v = rng.normal(size=300).astype(np.float32)
        return v / np.linalg.norm(v)


def install():
    import tarte_ai.tarte_utils as u
    import tarte_ai.tarte_preprocess_table as p
    f = lambda: StubFastText()
    u.load_fasttext = f
    p.load_fasttext = f
