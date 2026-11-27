# Troubleshooting: Transient `START=false` on GPU Nodes

## Problem
When querying GPU nodes on NMRbox (e.g., `matlab1.nmrbox.org`) via `condor_status -server`, users may see `START=false` and mistakenly conclude that the node is in maintenance or that GPU jobs are being rejected.

## Root Cause
This is usually a **transient state caused by slot retirement**, not a real drain. When another user's job finishes (or is evicted), HTCondor briefly marks the slot as retiring. During this window, `START=false` is reported even though the node is healthy and will accept new jobs as soon as the current job releases the slot.

## Diagnosis
Run the following to see whether the node is actually capable of running your job:

```bash
condor_q -better-analyze <JobID>
If the output contains a line like:

text
The following resources are untested:
  1 slots would match if drained
...it means the slot is in retirement, not in maintenance. The job will match as soon as the current occupant finishes.

Recommended GPU Targeting
Instead of hardcoding a specific machine name (fragile and creates hotspots), target GPUs by compute capability:

text
+Production = False
requirements = (GPUs_Capability >= 8)
This selects L40S (8.9) and H100 (9.0) GPUs while excluding older T4 (7.5) and V100 (7.0) cards. Verify the current GPU inventory on the NMRbox hardware page before relying on a specific capability level.

Best Practices
Do not request more GPUs than needed. Excessive requests lower your relative user priority.

Keep job runtimes under 8 hours. Break long simulations into checkpointed segments.

Use +Production = False only when your executable is self-contained (static binary or Python venv). Production nodes have the full NMR software stack; compute-only nodes do not.

Example: complete GPU submit file
bash
executable = /path/to/gromacs/bin/gmx
arguments  = mdrun -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu

log    = logs/md.log
output = logs/md.out
error  = logs/md.err

request_cpus   = 8
request_memory = 16GB
request_disk   = 10GB
request_gpus   = 1

# Target L40S (8.9) or H100 (9.0); excludes T4 (7.5) and V100 (7.0)
requirements = (GPUs_Capability >= 8)

+Production = False
should_transfer_files = NO
transfer_executable   = FALSE

queue
Related Issues
condor_status -server showing stale state: expected during slot retirement.

condor_q -better-analyze is the authoritative diagnostic; condor_status is a snapshot.

References
HTCondor Manual: Job Submission and Management

NMRbox GPU Hardware Page: https://nmrbox.org/
