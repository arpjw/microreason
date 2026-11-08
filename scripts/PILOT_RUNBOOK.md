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

## Known limits

- The ledger counts GPU time at the listed rate only. Storage and other charges are excluded,
  and the RunPod billing page is the source of truth for the actual bill.
- The pod scripts were exercised locally only in dry-run form (smoke model, MPS, fake pod id).
  The RunPod-specific steps (`RUNPOD_POD_ID`, self-termination, CUDA) are first exercised on
  the pod itself, which is why `pod_watchdog.sh check` runs before anything else.
