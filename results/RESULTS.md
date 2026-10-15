# Results


> **These numbers come from the synthetic test market, not real data.** Run steps 1-3 to replace them.


## All methods

| Method | Learned in | Spread it saw in practice | Clean train | Clean val | **Clean holdout** | Holdout reward (0-1) |
|---|---|---|---|---|---|---|
| v0: original Finagic | none | +19.1 pts | +19.9 pts | +32.1 pts | **+37.2 pts** | n/a |
| v1: bugs fixed | none | +21.0 pts | +23.1 pts | +31.7 pts | **+43.7 pts** | 0.50 |
| optimizer trained on clean | clean | +28.6 pts | +28.6 pts | +38.8 pts | **+54.1 pts** | 0.99 |
| optimizer trained on leaky | leaky | +32.9 pts | +22.9 pts | +41.3 pts | **+44.6 pts** | 0.73 |

## Fireworks RFT jobs

| RFT job | Practised in | Train reward: first → final | **Clean holdout reward: first → final** | Minutes | Cost |
|---|---|---|---|---|---|
| rft-clean | clean | pending → pending | **pending → pending** | pending | $0.00 |
| rft-leaky | leaky | pending → pending | **pending → pending** | pending | $0.00 |

## Leak ablation

| Only this leak switched on | Spread it saw in practice | Clean holdout spread |
|---|---|---|
| L1 timing (period end, not filing date) | +28.2 pts | +50.1 pts |
| L2 restated values | +28.4 pts | +49.7 pts |
| L3 price from the end of the return window | +29.4 pts | +41.1 pts |
| L4 today's index members | +31.2 pts | +56.0 pts |

## Inference

_pending: run step 6_