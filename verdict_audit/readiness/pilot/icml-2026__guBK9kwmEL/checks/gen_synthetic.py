"""Schema-faithful SYNTHETIC data for the RoVTL repository (no real data is read or downloaded).

Schema is taken from the repository's own loaders / attribute lists:
  * RoVTL/dataset/MRContrastiveDatasetH5.py, MRDownstreamDataset.py:
      HDF5 file with  images   : float array (N, 11 slices, 10 frames, H, W)   (paper: 11 x 10 x 128 x 128)
                      metadata : float array (N, n_cols) with attrs['columns'] = column names
      columns: 'eid', 'no_cad', every name in CATEGORICAL_ATTRIBUTES (values = keys of LABEL_MAPPING[col])
               and every name in NUMERICAL_ATTRIBUTES (floats); some NaN (missing) entries.
      label columns (multilabel, 4 outputs): I20_premr, I21v2_premr, I24_premr, I25_premr
  * a CSV next to each H5 with 'eid' (filter file) plus the raw attribute columns (used by the scripts to fit the
    TARTE table preprocessor).
Values are random. A planted latent structure is added so that mechanism checks have something to measure:
  z_shared (visible in image AND table), z_img (image only), z_tab (table only).
  Every table attribute is a noisy random mixture of [z_shared, z_tab]; the image is a noisy random mixture of
  [z_shared, z_img]; labels depend on all three latents (so the table carries information not visible in the image).
Usage: python gen_synthetic.py OUT_DIR N_TRAIN N_VAL N_TEST HW SEED [FRAMES=10 D_SHARED=6 D_TAB=3 TAB_NOISE=0.7 IMG_NOISE=0.5]
"""
import importlib.util
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

REPO_ATTR = "/work/work/repo/RoVTL/dataset/attributes.py"


def load_attributes(path=REPO_ATTR):
    spec = importlib.util.spec_from_file_location("rovtl_attributes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out_dir = sys.argv[1]
    n_train, n_val, n_test = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    hw = int(sys.argv[5])
    seed = int(sys.argv[6])
    frames = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    d_sh_arg = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    d_tab_arg = int(sys.argv[9]) if len(sys.argv) > 9 else 3
    tab_noise = float(sys.argv[10]) if len(sys.argv) > 10 else 0.7
    img_noise = float(sys.argv[11]) if len(sys.argv) > 11 else 0.5
    os.makedirs(out_dir, exist_ok=True)
    A = load_attributes()
    cats = list(A.CATEGORICAL_ATTRIBUTES)
    nums = list(A.NUMERICAL_ATTRIBUTES)
    assert len(set(cats)) == len(cats) and len(set(nums)) == len(nums) and not (set(cats) & set(nums))
    missing_maps = [c for c in cats if c not in A.LABEL_MAPPING]
    assert not missing_maps, f"categorical columns without LABEL_MAPPING: {missing_maps}"
    label_cols = ["I20_premr", "I21v2_premr", "I24_premr", "I25_premr"]
    assert all(c in cats for c in label_cols)

    rng = np.random.default_rng(seed)  # structure (mixing matrices) is shared by all splits
    d_sh, d_img, d_tab = d_sh_arg, 3, d_tab_arg
    # table mixing: each attribute sees shared+tab latents
    n_attr = len(cats) + len(nums)
    M_tab = rng.normal(size=(n_attr, d_sh + d_tab)) / np.sqrt(d_sh + d_tab)
    # image mixing: (11 slices x 10 frames) planes see shared+img latents
    M_img = rng.normal(size=(11, frames, d_sh + d_img)) / np.sqrt(d_sh + d_img)
    # label weights
    W_lab = rng.normal(size=(4, d_sh + d_img + d_tab))
    W_lab /= np.linalg.norm(W_lab, axis=1, keepdims=True)
    cat_keys = {c: sorted(A.LABEL_MAPPING[c].keys()) for c in cats}
    num_loc = {c: rng.normal(scale=5.0) + 50 for c in nums}
    num_scale = {c: abs(rng.normal(scale=5.0)) + 1.0 for c in nums}

    meta_cols = ["eid", "no_cad"] + cats + nums
    schema = {"columns": meta_cols, "n_cat": len(cats), "n_num": len(nums), "label_cols": label_cols,
              "images_shape_per_sample": [11, frames, hw, hw], "latents": {"shared": d_sh, "img": d_img, "tab": d_tab}}

    def make_split(name, n, eid0):
        r = np.random.default_rng(seed * 1000 + {"train": 1, "val": 2, "test": 3}[name])
        z_sh = r.normal(size=(n, d_sh)); z_im = r.normal(size=(n, d_img)); z_tb = r.normal(size=(n, d_tab))
        z_all = np.concatenate([z_sh, z_im, z_tb], axis=1)
        # labels
        logits = 2.2 * (z_all @ W_lab.T) * 1.0
        logits += r.normal(scale=0.3, size=logits.shape) - 0.4
        lab = (r.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(float)
        # table attributes
        zt = np.concatenate([z_sh, z_tb], axis=1) @ M_tab.T  # (n, n_attr)
        zt = zt + r.normal(scale=tab_noise, size=zt.shape)
        meta = np.full((n, len(meta_cols)), np.nan, dtype=np.float64)
        meta[:, 0] = eid0 + np.arange(n)
        for j, c in enumerate(cats):
            col = meta_cols.index(c)
            keys = cat_keys[c]
            if c in label_cols:
                meta[:, col] = lab[:, label_cols.index(c)]
                continue
            if len(keys) == 2 and set(keys) == {0, 1}:
                thr = np.quantile(zt[:, j], 0.8 if c.endswith("_premr") or c.endswith("_postmr") else 0.5)
                meta[:, col] = (zt[:, j] > thr).astype(float)
            else:
                edges = np.quantile(zt[:, j], np.linspace(0, 1, len(keys) + 1)[1:-1])
                idx = np.digitize(zt[:, j], edges)
                meta[:, col] = np.array(keys, dtype=float)[idx]
        for j, c in enumerate(nums):
            col = meta_cols.index(c)
            meta[:, col] = num_loc[c] + num_scale[c] * zt[:, len(cats) + j]
        # random missing entries in a few columns (UK Biobank-like sparsity), never in eid / labels
        miss = r.random((n, len(meta_cols))) < 0.03
        miss[:, [0, 1] + [meta_cols.index(c) for c in label_cols]] = False
        meta[miss] = np.nan
        meta[:, 1] = (lab.sum(axis=1) == 0).astype(float)  # no_cad
        # images
        zi = np.concatenate([z_sh, z_im], axis=1)
        planes = np.einsum("ctk,nk->nct", M_img, zi)  # (n, 11, frames)
        img = planes[:, :, :, None, None] * np.ones((1, 1, 1, hw, hw))
        gx = np.linspace(-1, 1, hw)[None, None, None, :, None]
        img = img + 0.3 * zi[:, 0, None, None, None, None] * gx  # a bit of spatial structure
        img = img + r.normal(scale=img_noise, size=img.shape)
        return meta.astype(np.float64), img.astype(np.float32), lab

    eid0 = 100000
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        meta, img, lab = make_split(name, n, eid0)
        eid0 += 100000
        with h5py.File(os.path.join(out_dir, f"{name}.h5"), "w") as f:
            f.create_dataset("images", data=img, compression=None)
            f.create_dataset("metadata", data=meta)
            f["metadata"].attrs.create("columns", np.array(meta_cols, dtype=object), dtype=h5py.string_dtype())
        df = pd.DataFrame(meta, columns=meta_cols)
        df["eid"] = df["eid"].astype(int)
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        print(f"{name}: images {img.shape} {img.dtype}  metadata {meta.shape}  label prevalence {lab.mean(0).round(2)} "
              f"nan_frac={np.isnan(meta).mean():.3f}")
    with open(os.path.join(out_dir, "schema.json"), "w") as fh:
        json.dump(schema, fh)


if __name__ == "__main__":
    main()
