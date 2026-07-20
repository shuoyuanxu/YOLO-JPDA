# YOLO-JPDA
<img width="640" height="360" alt="yolo-jpda-demo" src="https://github.com/user-attachments/assets/3cefd9c8-903d-4eb4-9fb4-9ca2eb129f56" />

Vibe-coded cleaned code for [*Real-time Implementation of YOLO+JPDA for Small
Scale UAV Multiple Object Tracking*](https://doi.org/10.1109/ICUAS.2018.8453398)
(ICUAS 2018). See [Citation](#citation).

## Run

Everything is driven by a YAML config. Edit `config/default.yaml`, then:

```bash
python main.py
```

## Modes

Set `mode:` in the config, or pass `--mode`.

| Mode | What it does | Needs a detector |
|---|---|---|
| `detect` | Detector only. Writes `detections.txt`. | yes |
| `track` | Detector + JPDA. The normal mode. Writes `tracked.mp4` and `tracks.txt`. | yes |
| `replay` | JPDA over a detection file. | no |
| `evaluate` | `replay`, then CLEAR-MOT metrics against ground truth. | no |

`replay` and `evaluate` need no deep-learning dependencies at all, which makes
them the fast way to tune the tracker or benchmark against MOT sequences.

## Layout

| Path | Contents |
|---|---|
| `main.py` | Entry point |
| [config/](config/) | YAML configs — `default.yaml`, `mot17.yaml` |
| [src/yolo_jpda/](src/yolo_jpda/) | The package |
| [src/yolo_jpda/tracking/](src/yolo_jpda/tracking/) | Kalman filter, gating, joint association, tracker |
| [src/yolo_jpda/detectors/](src/yolo_jpda/detectors/) | Detector backends behind one protocol |
| [src/yolo_jpda/io/](src/yolo_jpda/io/) | Image sequences, video, MOT-format results |
| `data/` | Put input sequences here (contents gitignored) |
| `model/` | Put detector weights here (contents gitignored) |

Sequences follow the MOT-challenge convention:

```
data/my_sequence/
├── img1/           # 000001.jpg, 000002.jpg, ...
├── det/det.txt     # frame, id, left, top, width, height, conf, -1, -1, -1
└── gt/gt.txt       # ground truth, for evaluate mode
```

## Detector backends

| Backend | Requires |
|---|---|
| `yolo` *(default)* | `ultralytics`; bare weight names like `yolov8n.pt` download on first use |

## Tuning

The parameters that matter most, and what going wrong looks like:

| Parameter | Effect | Symptom if wrong |
|---|---|---|
| `gate_confidence` | Validation-gate size | Too small: tracks starve and die, re-initiating every frame. Too large: ambiguous associations |
| `process_noise` | Trust in the motion model | Too small: lags behind manoeuvres. Too large: jitters, follows clutter |
| `measurement_noise_var` | Trust in the detector | Too small: overreacts to detector jitter. Too large: sluggish |
| `max_speed` | Fastest plausible target | Too small: fast targets are dropped on their first frame |
| `max_misses` | Coasting through occlusions | Too small: tracks break. Too large: ghosts linger |
| `min_hits` | Confirmation delay | Too small: clutter reported as objects |


## Citation

> S. Xu, A. Savvaris, S. He, H.-S. Shin and A. Tsourdos, "Real-time
> Implementation of YOLO+JPDA for Small Scale UAV Multiple Object Tracking,"
> *2018 International Conference on Unmanned Aircraft Systems (ICUAS)*,
> Dallas, TX, USA, 2018, pp. 1336–1341, doi: 10.1109/ICUAS.2018.8453398.

```bibtex
@inproceedings{xu2018yolojpda,
  title     = {Real-time Implementation of {YOLO}+{JPDA} for Small Scale {UAV}
               Multiple Object Tracking},
  author    = {Xu, Shuoyuan and Savvaris, Al and He, Shaoming and
               Shin, Hyo-Sang and Tsourdos, Antonios},
  booktitle = {2018 International Conference on Unmanned Aircraft Systems (ICUAS)},
  pages     = {1336--1341},
  year      = {2018},
  address   = {Dallas, TX, USA},
  publisher = {IEEE},
  doi       = {10.1109/ICUAS.2018.8453398}
}
```
