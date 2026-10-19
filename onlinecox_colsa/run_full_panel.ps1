$rscript = "C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
$env:R_LIBS_USER = "C:\Projects\02_academia\icml-repro\onlinecox_colsa\.rlib"
Set-Location "C:\Projects\02_academia\icml-repro\onlinecox_colsa"
$plan = @{ "6"="200"; "20"="120"; "50"="60"; "200"="25" }
foreach ($k in @("6","20","50","200")) {
  $env:K = $k; $env:REPS = $plan[$k]
  "===== K=$k REPS=$($plan[$k]) ====="
  & $rscript reproduce_claim2_full.R > "runs\claim2full_K$k.log" 2>&1
  Get-Content "runs\claim2full_K$k.log" -Tail 10
}
"ALL DONE"
