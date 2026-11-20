# biped_v1: averaging neighbouring MAC-41 checkpoints, M5 (MAC-42, WP4-34)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: d7b1ef9 (main), git_dirty false; eval only (no training)
method: average_checkpoints.py over k consecutive MAC-41 checkpoints (50 apart; k = 3 centres 6900–7550, k = 5 centres
        6950–7500), export via train_ppo --iters 0, eval_biped --terrain --robustness --dr --envs 16 --seconds 20 (STS3215
        curve, seed 0); seed-0 passes re-scored with --seed 1. Extra: k = 3 centred on 6850, which uses mac40d/model_6800,
        the checkpoint MAC-41 resumed from.
```

## All windows
(k, centre; flat \|wz\| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15; flat falls; 5 / 10 / 20 mm falls; P = passes on seed 0)
```
  k3 6850 (extra)  0.027   95 %  0  0/0/9   P  
  k3 6900  0.063   93 %  0  0/0/8   P  seed 1: 0.068 / 93 % / 0 / 0 -> pass
  k3 6950  0.070   97 %  0  0/0/7   P  seed 1: 0.078 / 97 % / 1 / 0 -> FAIL
  k3 7000  0.097   96 %  0  0/0/6      
  k3 7050  0.135   97 %  0  0/0/8      
  k3 7100  0.122   97 %  0  0/0/7      
  k3 7150  0.140   97 %  0  0/0/8      
  k3 7200  0.199   94 %  0  0/0/8      
  k3 7250  0.133   94 %  0  0/0/4      
  k3 7300  0.110   91 %  0  0/0/9      
  k3 7350  0.110   94 %  0  0/0/7      
  k3 7400  0.096   93 %  0  0/0/9      
  k3 7450  0.068   94 %  0  0/0/8   P  seed 1: 0.078 / 94 % / 0 / 0 -> pass
  k3 7500  0.112  101 %  1  0/0/7      
  k3 7550  0.224  101 %  0  0/0/10     
  k5 6950  0.056   93 %  0  0/0/6   P  seed 1: 0.066 / 93 % / 0 / 0 -> pass
  k5 7000  0.120   97 %  0  0/0/9      
  k5 7050  0.095   96 %  0  0/0/7      
  k5 7100  0.136   97 %  0  0/0/7      
  k5 7150  0.166   96 %  0  0/0/8      
  k5 7200  0.106   96 %  0  0/0/8      
  k5 7250  0.063   93 %  0  0/0/9   P  seed 1: 0.049 / 93 % / 0 / 0 -> pass
  k5 7300  0.108   93 %  0  0/0/5      
  k5 7350  0.090   93 %  0  0/0/7      
  k5 7400  0.109   94 %  0  0/0/10     
  k5 7450  0.085   97 %  1  0/0/2      
  k5 7500  0.177   98 %  0  0/0/9      
```

## Verdict
**Met.** The k = 3 average centred on 6900 (MAC-41 models 6850, 6900 and 6950) passes the WP4-32 rows on seed 0
(|wz| 0.063, 40 ms 93 %, 0 flat falls, 10 mm 0/16) and on seed 1 (0.068, 93 %, 0, 0/16). Its same-k neighbours,
6850 (0.027, extra) and 6950 (0.070), also pass on seed 0, so three windows in a row pass.

Seed-0 flat rows: 0.15 / 0.25 with DR off give vx 0.164 / 0.238, wz −0.063 / −0.005. With DR on they give vx 0.148 / 0.224,
wz −0.028 / −0.009. The zero command gives |wz| 0.051, turn 0.8 gives 0.491, and 40 ms keeps 0.139. On 10 mm tiles vx is 0.149.

**Terrain policy candidate: k = 3, centre 6900, ONNX sha256 `2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3`**
(not committed).

Other seed-0 passes fail the extra checks. k = 3 6950 fails seed 1 with one flat fall. k = 3 7450, k = 5 6950 and k = 5 7250 pass both
seeds, but a same-k neighbour fails. Averaging narrows the yaw swing (k = 3: 0.063–0.224, against 0.078–0.527 for single
checkpoints) but does not remove it. Caveat: the −50 neighbour of 6900 contains the run's start checkpoint.

Files: `mac42/k<k>_<centre>-eval.json` for all 27 windows, `…-eval-seed1.json` for the five that passed seed 0.
