# Claim 1: fixed ECG/VCG geometry


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e711bb617d0e", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 1: fixed ECG/VCG geometry"}
-->
**PARTIAL - 1/2.** The printed ridge lift and projection recover
synthetic latent trajectories with maximum error
`6.90e-06`. Official execution is
blocked because the geometry module imports absent `lvcg.data` modules.


---
<!-- trackio-cell
{"type": "code", "id": "cell_816678096e00", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 1: fixed ECG/VCG geometry evidence", "language": "python"}
-->
````output
{
  "geometry": {
    "seed": 260531249,
    "shape": {
      "directions": [
        4,
        5,
        3
      ],
      "visible": [
        4,
        5,
        128
      ],
      "latent": [
        4,
        3,
        128
      ]
    },
    "max_latent_abs_error": 6.895039269760872e-06,
    "max_visible_reprojection_abs_error": 3.7422574732381975e-06,
    "passes_1e_4": true
  },
  "runtime": {
    "command": "import lvcg.models.vcg; from lvcg.models.lvcg import LVCG",
    "returncode": 1,
    "succeeded": false,
    "last_error_line": "ModuleNotFoundError: No module named 'lvcg.data'",
    "missing_relative_imports": [
      {
        "file": "lvcg/models/lvcg.py",
        "line": 68,
        "import": "..data.angle",
        "resolved": "lvcg/data/angle"
      },
      {
        "file": "lvcg/models/lvcg.py",
        "line": 69,
        "import": "..data.beat_segmentation",
        "resolved": "lvcg/data/beat_segmentation"
      },
      {
        "file": "lvcg/models/vcg.py",
        "line": 20,
        "import": "..data.angle",
        "resolved": "lvcg/data/angle"
      }
    ]
  }
}
````
