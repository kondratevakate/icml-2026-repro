# Claim 2: safety-generalization gap


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_14bb174862bb", "created_at": "2026-07-29T20:21:46+00:00", "title": "Claim 2: safety-generalization gap"}
-->
**Verdict - PARTIALLY VERIFIED IN ONE SCOPED RUN (1/2).**

A released CPO/T1D/adolescent/seed-0 checkpoint was reconstructed independently
from its config and state dictionary. One seven-day rollout gives ID TIR
`100.00%` and risk index
`1.12`. Across patients 2-10, mean TIR is
`89.00%` and risk index is
`3.10`: gaps of
`-11.00` percentage points and
`+1.99`.

The metric trace records post-step glucose. The released evaluator requests
`info["cgm"]`, but GlucoSim omits that key and the released code falls back to
the pre-step observation. This is therefore a corrected-protocol mechanism
check, not an exact replay of that fallback or the paper's three-seed,
eight-algorithm aggregate.


---
<!-- trackio-cell
{"type": "code", "id": "cell_127b5016d087", "created_at": "2026-07-29T20:21:46+00:00", "title": "C2 machine-readable evidence", "language": "json"}
-->
````output
{
  "checkpoint_sha256": "806fe99618dc15549eb3a543b1464329f538f30cbb6ca0eecc8ab07058dac1c8",
  "evaluation_seed": 20260729,
  "id": {
    "action_sha256": "5644c35d654d275032ff9a87fc3d1cc66f5473ab8560ab520c742f07cacabf3e",
    "cost_sum": 233.77151572399322,
    "cv_percent": 7.027454129367715,
    "maximum_glucose": 164.5228271484375,
    "minimum_glucose": 105.31069946289062,
    "patient": "adolescent#001",
    "reward_sum": 136.50327868717625,
    "risk_index": 1.1165919506828756,
    "steps": 2016,
    "tir_percent": 100.0
  },
  "metric_trace_contract": "CORRECTED_POST_STEP_OBSERVATION_0_ONE_VALUE_PER_ACTION",
  "ood": [
    {
      "action_sha256": "4e3dce4658a9dee4d2d1c5d6acf060bd1a6d9a143a983edb5bbbf792d482774f",
      "cost_sum": 1823.820402911359,
      "cv_percent": 28.250851527062885,
      "maximum_glucose": 400.0,
      "minimum_glucose": 113.39875030517578,
      "patient": "adolescent#002",
      "reward_sum": -81.25708136597348,
      "risk_index": 5.142082499141964,
      "steps": 2016,
      "tir_percent": 77.43055555555556
    },
    {
      "action_sha256": "9c4752b90f660a46ad009d55cb288b38082b5688d8a9e5296b5b63a8faeee905",
      "cost_sum": 3906.737057873394,
      "cv_percent": 13.11937073915022,
      "maximum_glucose": 180.78775024414062,
      "minimum_glucose": 90.63156127929688,
      "patient": "adolescent#003",
      "reward_sum": 191.34762181292365,
      "risk_index": 0.7233785321577461,
      "steps": 2016,
      "tir_percent": 99.8015873015873
    },
    {
      "action_sha256": "5277b5a276258a7b2677467ebafcb82d8816b5282aa0b14ee73393a517b93cd2",
      "cost_sum": 436.8316771309571,
      "cv_percent": 16.160252640147565,
      "maximum_glucose": 193.28536987304688,
      "minimum_glucose": 88.98251342773438,
      "patient": "adolescent#004",
      "reward_sum": 174.09350277538954,
      "risk_index": 0.9222641364534663,
      "steps": 2016,
      "tir_percent": 98.75992063492063
    },
    {
      "action_sha256": "d3a914f43474bb355e7b4962764dd0dea0fc7dc153460ec2e082d4ce5c510d6f",
      "cost_sum": 309.2218309079356,
      "cv_percent": 17.07474724202811,
      "maximum_glucose": 223.50035095214844,
      "minimum_glucose": 112.14228057861328,
      "patient": "adolescent#005",
      "reward_sum": -66.83554255184242,
      "risk_index": 3.0504844385763077,
      "steps": 2016,
      "tir_percent": 89.13690476190477
    },
    {
      "action_sha256": "b7199bba17d8d75f0733e3f28b8d4c3ec561c1ef730e9ee35b5b134da5bcb16b",
      "cost_sum": 5457.100023811066,
      "cv_percent": 14.379270819238306,
      "maximum_glucose": 204.6819305419922,
      "minimum_glucose": 106.39045715332031,
      "patient": "adolescent#006",
      "reward_sum": 173.93101896943892,
      "risk_index": 1.5903611651494172,
      "steps": 2016,
      "tir_percent": 97.27182539682539
    },
    {
      "action_sha256": "53e9ea71c0a30b2eb4b82ca4e305452a7a42ef2b63a8d27c2f0464a62d891b1d",
      "cost_sum": 6773.203966939984,
      "cv_percent": 20.429149106279286,
      "maximum_glucose": 251.4269256591797,
      "minimum_glucose": 107.2674331665039,
      "patient": "adolescent#007",
      "reward_sum": 63.96515428190662,
      "risk_index": 2.889860998977157,
      "steps": 2016,
      "tir_percent": 85.06944444444444
    },
    {
      "action_sha256": "79c6134d3f847920092d06c823e2deacba7f39b6ccf1bdf1328d5ccb2f9098fa",
      "cost_sum": 2887.251149393618,
      "cv_percent": 38.40822083059688,
      "maximum_glucose": 400.0,
      "minimum_glucose": 132.25413513183594,
      "patient": "adolescent#008",
      "reward_sum": -3718.7316067786055,
      "risk_index": 12.336851721206726,
      "steps": 165,
      "tir_percent": 55.15151515151515
    },
    {
      "action_sha256": "efb7101014b1b500bdb0a7a10439704cf8fbf97f7f83324dc2bf6183cd3c7dcb",
      "cost_sum": 173.89780020307467,
      "cv_percent": 11.299425674324258,
      "maximum_glucose": 168.44546508789062,
      "minimum_glucose": 77.04744720458984,
      "patient": "adolescent#009",
      "reward_sum": 255.59594172942354,
      "risk_index": 0.47748536666378066,
      "steps": 2016,
      "tir_percent": 100.0
    },
    {
      "action_sha256": "cd1803bebecff7ed11ceb9d056da4a1bcae81b397198b6ee3fa4ff885ad88a25",
      "cost_sum": 150.51920821777034,
      "cv_percent": 13.683667193197014,
      "maximum_glucose": 186.05075073242188,
      "minimum_glucose": 98.69580078125,
      "patient": "adolescent#010",
      "reward_sum": 154.78461233790188,
      "risk_index": 0.7987263749186432,
      "steps": 2016,
      "tir_percent": 98.4126984126984
    }
  ],
  "ood_mean_risk_index": 3.1034994703605787,
  "ood_mean_tir_percent": 89.00382796216131,
  "released_evaluator_effective_trace": "PRE_STEP_OBSERVATION_0_BECAUSE_INFO_HAS_NO_CGM_KEY",
  "risk_gap_ood_minus_id": 1.986907519677703,
  "scope": "CPO_T1D_ADOLESCENT_SEED0_ONE_7_DAY_EPISODE_PER_PATIENT",
  "tir_gap_ood_minus_id": -10.99617203783869
}
````
