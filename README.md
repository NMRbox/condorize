# condorize

A command-line tool that monitors a program's resource usage and generates an
HTCondor submit file. Designed for NMRbox users who want to run jobs on the
HTCondor pool without writing submit files by hand.

## Usage

```
condorize [--timeout SECONDS] [--output FILE] -- command [arguments...]
```

Prepend `condorize --` to the command you would normally run:

```
condorize -- voronota-voromqa --input structure.pdb
```

To change the monitoring duration from the default of 60 seconds:

```
condorize --timeout 120 -- my_slow_program arg1 arg2
```

To specify the output submit file path:

```
condorize --output myjob.sub -- myprogram arg1
```

## What it does

### 1. Monitors your program

Condorize launches your command and monitors it for the specified duration
(default 60 seconds), sampling every 0.5 seconds. It tracks:

- **Memory** (RSS) across the entire process tree (parent and all child
  processes), read directly from `/proc`.
- **CPU/thread count** across the process tree, to determine how many CPUs
  your job needs.
- **GPU usage** using two methods: checking `/proc/PID/fd` for open
  `/dev/nvidia*` file descriptors, and querying `nvidia-smi` for active
  compute processes.

After the timeout, the process is terminated with SIGTERM (and SIGKILL if it
doesn't exit within 5 seconds). If the process exits on its own before the
timeout, monitoring stops early.

A live status line shows progress during monitoring:

```
  [15s/60s] Peak RSS: 245.3 MB | CPUs: 4 | GPU: No
```

If memory usage is still increasing when monitoring stops, condorize will warn
you that the observed values may be underestimates and suggest running with a
longer `--timeout`.

If the program exits quickly with a non-zero exit code (e.g., bad arguments
or a missing input file), condorize will warn you and ask whether to continue,
since the monitored resource usage likely doesn't reflect the program's real
needs.

### 2. Inspects the package (in parallel)

While the program is being monitored, condorize looks up the executable's
package in the background:

- Resolves the binary path using `which` and follows symlinks.
- Runs `dpkg -S` to find which package owns the binary.
- Checks the package for `Nmrbox-Software` and `Nmrbox-Version` metadata
  using `dpkg-query`. If found, it formats an HTCondor requirement to ensure
  the job runs on a node with the correct software version installed (e.g.,
  `requirements = VORONOTA == "v121"`).

### 3. Asks you to confirm

After monitoring, condorize displays a summary of what it detected and
prompts you to confirm or adjust each value. Press Enter to accept the
suggested default (shown in brackets), or type a new value:

```
============================================================
  Condorize - Detected Settings
============================================================
  Peak memory (RSS):  245.3 MB
  Suggested request:  320 MB (with 25% headroom)
  Peak CPUs/threads:  4
  GPU used:           No
  NMRBox requirement: VORONOTA == "v121"
============================================================

  Review the settings below. Press Enter to accept the
  suggested value shown in [brackets], or type a new value.

  Memory to request in MB [320]:
  CPUs to request [4]:
  Request a GPU? [y/N]:
  Include NMRBox requirement 'VORONOTA == "v121"'? [Y/n]:
```

Memory suggestions include 25% headroom over the observed peak, rounded up to
the nearest 64 MB.

If an NMRBox requirement is included, condorize will query `condor_status` to
check whether any machines in the pool currently match. If none do, it will
print a warning so you can investigate before submitting.

### 4. Writes the submit file

Condorize writes a `.sub` file named after the executable (e.g.,
`voronota-voromqa.sub`) in the current directory, ready to submit. If a file
with that name already exists, you will be asked whether to overwrite it or
choose a different name.

```
condor_submit voronota-voromqa.sub
```

The generated submit file includes:

- `executable` and `arguments` (properly quoted for HTCondor)
- `initialdir` set to the directory where you ran condorize
- `request_memory`, `request_cpus`, `request_disk` (default 2 GB), and
  `request_gpus` (if needed)
- `requirements` for NMRBox software version (if applicable)
- `+Production = True` to target production NMRbox machines
- `output`, `error`, and `log` files named with cluster and process IDs
- `getenv = True` to preserve your shell environment
- `should_transfer_files = NO` for the shared filesystem (automatically
  switches to `IF_NEEDED` if paths on `/tmp` or `/scratch` are detected)

## File transfer detection

Condorize checks the executable path, current working directory, and all
command arguments that look like file paths. If any of them are located under
`/tmp` or `/scratch` (which are local to each machine and not shared),
it sets `should_transfer_files = IF_NEEDED` so HTCondor will handle file
transfers. Otherwise it uses `should_transfer_files = NO` to rely on the
shared filesystem.

## Requirements

- Python 3.6+
- Linux (uses `/proc` filesystem for monitoring)
- `dpkg` and `dpkg-query` (for package inspection; gracefully skipped if unavailable)
- `nvidia-smi` (for GPU detection; gracefully skipped if unavailable)
- `condor_status` (for pool validation; gracefully skipped if unavailable)
