# Claim 2: aligned-constraint mechanics


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f80bf75eb920", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 2: aligned-constraint mechanics"}
-->
**VERIFIED - 2/2.** The official linear offset equals an
independent closed-form calculation exactly, is detached, and the delay/ramp
schedule matches all boundary cases. With zero zeroth-order coefficients, the
unguarded denominator produces NaN.


---
<!-- trackio-cell
{"type": "code", "id": "cell_b91d35f4e683", "created_at": "2026-07-30T07:41:34+00:00", "title": "Claim 2: aligned-constraint mechanics evidence", "language": "python"}
-->
````output
{
  "aligned_constraint": {
    "seed": 260525001,
    "expected_c": -0.33486423514651853,
    "official_c": -0.33486423514651853,
    "absolute_difference": 0.0,
    "offset_detached": true,
    "schedule": {
      "0": 0.0,
      "1": 0.0,
      "2": 0.0,
      "3": 0.3333333333333333,
      "4": 0.6666666666666666,
      "5": 1.0,
      "6": 1.0
    },
    "expected_schedule": {
      "0": 0.0,
      "1": 0.0,
      "2": 0.0,
      "3": 0.3333333333333333,
      "4": 0.6666666666666666,
      "5": 1.0,
      "6": 1.0
    },
    "total_is_finite": true
  },
  "degenerate": {
    "case": "all zeroth-order coefficients are zero",
    "official_c": NaN,
    "is_finite": false,
    "denominator_guard_present": false
  }
}
````
