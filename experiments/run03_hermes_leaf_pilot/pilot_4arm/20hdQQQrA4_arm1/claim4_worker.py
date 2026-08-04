"""claim4_worker.py <kind> <seed> <epochs> -> results/claim4_<kind>_<seed>.json"""
import json, sys, time
import verify_claim4 as V

kind, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
r = V.train(kind, seed, epochs=epochs)
r.update(kind=kind, epochs=epochs, seconds=round(time.time() - t0, 1))
json.dump(r, open(f"results/claim4_{kind}_{seed}.json", "w"), indent=2)
print(json.dumps(r))
