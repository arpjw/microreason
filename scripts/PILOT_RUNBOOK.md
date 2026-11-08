# Phase 5 GPU pilot runbook

Nothing here may start until the project owner approves, in chat, a specific provider, GPU,
listed hourly rate, dollar cap, runtime limit and shutdown rule. Re-read the live price
immediately before creating the pod. If it differs from the approved rate, stop and ask again.

## 0. Before approval (no cost)

- `uv run pytest` passes locally and the working tree is committed (`pod_push.sh` refuses
  uncommitted changes).
- `scripts/expected_fingerprints.json` is current (`uv run python scripts/check_fingerprints.py`).
- An SSH public key is registered on the RunPod account. Registering one changes account
  settings, so it needs its own approval.

## 1. Record the approval

Edit `configs/pilot_43m.yaml`: set `hourly_rate_usd`, `max_spend_usd`, `max_runtime_hours`,
`approved_by` and `approved_at` to the approved values, then commit. Until then the config fails
validation, and every paid run refuses to start.

## 1b. Watchdog key (first attempt finding)

In the first pilot attempt (2026-11-08, pod `5gqf57qzyj9ck6`, about 3 minutes, roughly $0.05) the
pod's built-in `RUNPOD_API_KEY` was refused with 403, even for reading its own pod. The pod was
therefore deleted before any work, as this runbook requires. The pod also ships runpodctl 1.14,
which uses different commands; the watchdog now installs the checksum-verified v2.15.0.

Self-termination needs a key that can manage pods, available inside the pod. The owner creates
it in the RunPod console and stores it as a RunPod Secret. Its value never passes through the
agent, a config file or a log. The pod is created with
`env: {"MICROREASON_WATCHDOG_KEY": "{{ RUNPOD_SECRET_<name> }}"}` and the watchdog uses it.

## 2. Create the pod

On-demand, 1× the approved GPU, the approved cloud, SSH enabled, an official RunPod PyTorch image
(it provides sshd and runpodctl), and a 40 GB container disk. Work stays on the container disk
(`/root/microreason`), because `/workspace` can be slow network storage. Record the pod id and
creation time. **Billing starts at creation.**

## 3. Arm the watchdog first

```bash
scripts/pod_push.sh root@<host> <port>
ssh -p <port> root@<host> 'cd /root/microreason && scripts/pod_watchdog.sh check'
ssh -p <port> root@<host> "cd /root/microreason && scripts/pod_watchdog.sh arm <created_epoch + ceiling_seconds>"
```

`ceiling_seconds` is `min(max_runtime_hours, max_spend_usd / hourly_rate_usd) × 3600`. If
`check` fails, the pod cannot terminate itself: delete it immediately and stop.

## 4. Setup and pilot

```bash
ssh -p <port> root@<host> 'cd /root/microreason && scripts/pod_setup.sh'
ssh -p <port> root@<host> "cd /root/microreason && MICROREASON_BILLING_START=<created_iso> scripts/pod_pilot.sh"
```

`pod_setup.sh` rebuilds all data from pinned sources and must print `fingerprints match`. Every
full-model run is guarded. It stops `reserve_minutes` before the tighter ceiling, saves a
checkpoint, and writes `budget_ledger.json`.

## 5. Copy results off, then terminate

```bash
scripts/pod_pull.sh root@<host> <port> <pod_id>
```

Then delete (terminate) the pod. Do not stop it, because a stopped pod keeps billing for its
volume disk. Confirm with a pod listing that it is gone. Terminate on any failure as well.
Results land in `runs/pod_results/<pod_id>/`. The summary is `reports/pilot/pilot_v1.json`
inside it.

## Lessons from pilot v1 (2026-11-08)

- RTX 4090 Secure Cloud capacity was scarce. Get approval for fallback GPUs and regions up front.
- The container sees all host CPUs. `pod_env.sh` caps thread pools at the cgroup quota; without
  the cap, pytest stalled for more than 30 minutes.
- Use micro-batch 16 for the 43M model. It was the fastest setting, at 7.6 GiB.

- When uploading a checkpoint from macOS, set `COPYFILE_DISABLE=1` on `tar`. Otherwise macOS adds
  `._*` AppleDouble files, and checkpoint verification correctly rejects them (SFT session,
  2026-11-08).

## Known limits

- The ledger counts GPU time at the listed rate only. Storage and other charges are excluded,
  and the RunPod billing page is the source of truth for the actual bill.
- The pod scripts were exercised locally only in dry-run form (smoke model, MPS, fake pod id).
  The RunPod-specific steps (`RUNPOD_POD_ID`, self-termination, CUDA) are first exercised on
  the pod itself, which is why `pod_watchdog.sh check` runs before anything else.
