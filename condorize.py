#!/usr/bin/env python3
"""condorize - Monitor a program's resource usage and generate an HTCondor submit file."""

import argparse
import math
import os
import subprocess
import shutil
import sys
import threading
import time

__version__ = "1.0.1"


def parse_args():
    """Parse command line arguments, splitting on '--' to separate our flags from the target command."""
    argv = sys.argv[1:]

    if "--" in argv:
        idx = argv.index("--")
        our_args = argv[:idx]
        cmd_args = argv[idx + 1:]
    else:
        # Use argparse's parse_known_args to consume our flags and leave the rest
        # as the command. This avoids needing to manually track which flags take values.
        parser = _build_parser()
        args, cmd_args = parser.parse_known_args(argv)
        if not cmd_args:
            parser.error("No command specified. Usage: condorize [--timeout N] -- command [args...]")
        return args.timeout, args.output, cmd_args

    parser = _build_parser()
    args = parser.parse_args(our_args)

    if not cmd_args:
        parser.error("No command specified. Usage: condorize [--timeout N] -- command [args...]")

    return args.timeout, args.output, cmd_args


def _build_parser():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="condorize",
        description="Monitor a command and generate an HTCondor submit file.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Monitoring duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output submit file path (default: <command>.sub)",
    )
    return parser


def get_direct_children(pid):
    """Get direct child PIDs of a process by walking /proc."""
    children = set()
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    stat = f.read().split()
                    # Field 4 (0-indexed 3) is PPID
                    ppid = int(stat[3])
                    if ppid == pid:
                        children.add(int(entry))
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                continue
    except FileNotFoundError:
        pass
    return children


def get_process_tree_pids(root_pid):
    """Get root PID plus all descendants via breadth-first traversal."""
    pids = {root_pid}
    frontier = [root_pid]
    while frontier:
        next_frontier = []
        for pid in frontier:
            children = get_direct_children(pid)
            for child in children:
                if child not in pids:
                    pids.add(child)
                    next_frontier.append(child)
        frontier = next_frontier
    return pids


def read_proc_memory(pid):
    """Read VmRSS from /proc/PID/status. Returns RSS in KB, or 0 on failure."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # value is in kB
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        pass
    return 0


def read_proc_threads(pid):
    """Read thread count from /proc/PID/status. Returns thread count, or 0 on failure."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        pass
    return 0


def read_proc_cpu_ticks(pid):
    """Read total CPU ticks (utime + stime) from /proc/PID/stat.
    Returns ticks consumed, or 0 on failure."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        # The comm field (field 2) is in parens and may contain spaces/parens.
        # Find the last ')' to reliably skip past it.
        i = data.rfind(')')
        if i < 0:
            return 0
        fields = data[i + 2:].split()
        # After ')': state(0), ppid(1), ..., utime(11), stime(12)
        utime = int(fields[11])
        stime = int(fields[12])
        return utime + stime
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return 0


def check_gpu_fd(pid):
    """Check if process has /dev/nvidia* file descriptors open."""
    try:
        fd_dir = f"/proc/{pid}/fd"
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
                if "/dev/nvidia" in target:
                    return True
            except (FileNotFoundError, PermissionError):
                continue
    except (FileNotFoundError, PermissionError):
        pass
    return False


def check_gpu_nvidia_smi(pids):
    """Check nvidia-smi for GPU usage by any of the given PIDs.
    Returns (gpu_used: bool, gpu_memory_mb: int)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False, 0

        max_gpu_mem = 0
        found = False
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    smi_pid = int(parts[0])
                    gpu_mem = int(parts[1])
                    if smi_pid in pids:
                        found = True
                        max_gpu_mem = max(max_gpu_mem, gpu_mem)
                except ValueError:
                    continue
        return found, max_gpu_mem
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, 0


def collect_open_files(pids):
    """Collect file paths opened by the given PIDs from /proc maps and fd."""
    files = set()
    for pid in pids:
        # Memory-mapped files (shared libraries, executables)
        try:
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6:
                        path = parts[-1]
                        if path.startswith('/'):
                            files.add(path)
        except (FileNotFoundError, PermissionError):
            pass
        # Open file descriptors
        try:
            fd_dir = f"/proc/{pid}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                    if target.startswith('/'):
                        files.add(target)
                except (FileNotFoundError, PermissionError):
                    continue
        except (FileNotFoundError, PermissionError):
            pass
    return files


def _file_scanner(root_pid, stop_event, all_open_files):
    """Background thread: rapidly scan process tree for open files.

    Short-lived child processes (e.g. commands in a shell script) may start and
    exit between the main monitoring loop's 0.5 s samples.  This thread polls at
    50 ms intervals, reading each descendant's /proc/PID/exe (a single readlink,
    very fast) so we catch executables that would otherwise be missed.  A full
    maps + fd collection runs less frequently to capture shared libraries and
    data files without excessive overhead.
    """
    iteration = 0
    while not stop_event.is_set():
        try:
            pids = get_process_tree_pids(root_pid)
            for pid in pids:
                # Fast: read /proc/PID/exe (catches compiled binaries)
                try:
                    exe = os.readlink(f"/proc/{pid}/exe")
                    if exe.startswith('/'):
                        all_open_files.add(exe)
                except (FileNotFoundError, PermissionError, OSError):
                    pass
                # Fast: read /proc/PID/cmdline (catches script files —
                # when bash runs a script, exe points to the interpreter
                # but cmdline contains the script path)
                try:
                    with open(f"/proc/{pid}/cmdline", 'rb') as f:
                        for arg in f.read().split(b'\0'):
                            try:
                                arg = arg.decode('utf-8', errors='replace')
                            except Exception:
                                continue
                            if arg.startswith('/'):
                                all_open_files.add(arg)
                except (FileNotFoundError, PermissionError, OSError):
                    pass
            # Slower: full maps + fd scan every ~1 s (20 iterations * 50 ms)
            iteration += 1
            if iteration % 20 == 0:
                all_open_files.update(collect_open_files(pids))
        except Exception:
            pass
        stop_event.wait(0.05)


def monitor_process(cmd, timeout):
    """Run cmd for up to timeout seconds, monitoring memory, CPU, and GPU usage.
    Returns (peak_rss_kb, avg_cpus, gpu_used, gpu_memory_mb, still_climbing, exited_early, exit_code, all_open_files)."""
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied: {cmd[0]}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting: {' '.join(cmd)}")
    print(f"Monitoring for up to {timeout} seconds...")
    pid = proc.pid

    peak_rss_kb = 0
    gpu_used = False
    gpu_memory_mb = 0
    exited_early = False
    exit_code = None
    all_open_files = set()
    # Track recent memory samples to detect if usage is still climbing
    recent_rss = []
    # Track CPU ticks to compute average CPU utilization
    clock_ticks_per_sec = os.sysconf('SC_CLK_TCK')
    total_cpu_ticks = 0
    prev_ticks = {}  # pid -> last known ticks
    first_sample = True
    start = time.time()

    # Start background file scanner to catch short-lived child processes
    scanner_stop = threading.Event()
    scanner_thread = threading.Thread(
        target=_file_scanner, args=(pid, scanner_stop, all_open_files), daemon=True,
    )
    scanner_thread.start()

    try:
        while time.time() - start < timeout:
            # Collect PIDs in the process tree
            pids = get_process_tree_pids(pid)

            # Check if process already exited
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start
                print(f"\r  Process exited on its own (code {ret}) after {elapsed:.1f}s"
                      + " " * 20)
                exited_early = True
                exit_code = ret
                break

            # Memory: sum RSS across process tree
            total_rss = sum(read_proc_memory(p) for p in pids)
            peak_rss_kb = max(peak_rss_kb, total_rss)
            recent_rss.append(total_rss)
            # Keep last 10 samples (5 seconds worth)
            if len(recent_rss) > 10:
                recent_rss.pop(0)

            # CPUs: track cumulative CPU ticks across the process tree.
            # On each sample we compute the delta in ticks per process;
            # for new processes we count their full accumulated ticks
            # (they just spawned, so this approximates time since last sample).
            current_ticks = {}
            for p in pids:
                current_ticks[p] = read_proc_cpu_ticks(p)
            if first_sample:
                # Record baseline; don't count any ticks yet
                first_sample = False
            else:
                for p, ticks in current_ticks.items():
                    if p in prev_ticks:
                        total_cpu_ticks += max(0, ticks - prev_ticks[p])
                    else:
                        # New process - count all its accumulated ticks
                        total_cpu_ticks += ticks
            prev_ticks = current_ticks

            # GPU: check file descriptors
            if not gpu_used:
                for p in pids:
                    if check_gpu_fd(p):
                        gpu_used = True
                        break

            # GPU: check nvidia-smi
            smi_used, smi_mem = check_gpu_nvidia_smi(pids)
            if smi_used:
                gpu_used = True
                gpu_memory_mb = max(gpu_memory_mb, smi_mem)

            elapsed = time.time() - start
            rss_mb = peak_rss_kb / 1024
            if elapsed > 0:
                avg_cpus = total_cpu_ticks / (clock_ticks_per_sec * elapsed)
            else:
                avg_cpus = 0
            sys.stdout.write(
                f"\r  [{elapsed:.0f}s/{timeout}s] Peak RSS: {rss_mb:.1f} MB | "
                f"Avg CPUs: {avg_cpus:.1f} | "
                f"GPU: {'Yes' if gpu_used else 'No'}"
                f"{f' ({gpu_memory_mb} MB)' if gpu_memory_mb else ''}   "
            )
            sys.stdout.flush()

            time.sleep(0.5)
        else:
            # Timeout reached - terminate
            print(f"\n\nTimeout reached ({timeout}s). Terminating process...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not exit after SIGTERM, sending SIGKILL...")
                proc.kill()
                proc.wait(timeout=5)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Stop the file scanner
    scanner_stop.set()
    scanner_thread.join(timeout=2)

    print()

    # Detect if resource usage was still climbing when we stopped
    still_climbing = False
    if not exited_early and len(recent_rss) >= 6:
        first_half = recent_rss[:len(recent_rss) // 2]
        second_half = recent_rss[len(recent_rss) // 2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        # If the second half average is >10% higher than the first half, usage is climbing
        if avg_first > 0 and (avg_second - avg_first) / avg_first > 0.10:
            still_climbing = True

    # Compute average CPUs used over the monitoring period
    elapsed = time.time() - start
    if elapsed > 0 and total_cpu_ticks > 0:
        avg_cpus = total_cpu_ticks / (clock_ticks_per_sec * elapsed)
    else:
        avg_cpus = 1
    suggested_cpus = max(1, int(math.ceil(avg_cpus)))

    return peak_rss_kb, suggested_cpus, gpu_used, gpu_memory_mb, still_climbing, exited_early, exit_code, all_open_files


def precompute_nmrbox_data():
    """Build a mapping of all installed NMRBox packages and a reverse dependency map.

    Returns (nmrbox_pkgs, dep_to_nmrbox) where:
      nmrbox_pkgs: {package_name: (software, version)}
      dep_to_nmrbox: {dependency_name: {(software, version), ...}}

    Safe to call from a background thread.
    """
    nmrbox_pkgs = {}

    # Get all installed packages with NMRBox metadata in one call
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f",
             "${Package}\t${Nmrbox-Software}\t${Nmrbox-Version}\n"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split('\t')
                if len(parts) >= 3 and parts[1].strip() and parts[2].strip():
                    nmrbox_pkgs[parts[0].strip()] = (parts[1].strip(), parts[2].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Build reverse mapping: dependency -> set of (software, version) from NMRBox packages.
    # This handles the case where a binary lives in a plain upstream package that was
    # pulled in as a dependency of an nmrbox-tool-* wrapper.
    dep_to_nmrbox = {}
    for pkg, (sw, ver) in nmrbox_pkgs.items():
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f", "${Depends}\n${Pre-Depends}\n", pkg],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                continue
            for line in result.stdout.splitlines():
                for dep_spec in line.split(','):
                    dep_name = dep_spec.strip().split('(')[0].split('|')[0].strip()
                    if dep_name:
                        dep_to_nmrbox.setdefault(dep_name, set()).add((sw, ver))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return nmrbox_pkgs, dep_to_nmrbox


def _find_owning_packages(files):
    """Find which dpkg packages own the given files. Returns a set of package names."""
    # Filter to paths that could plausibly be package-managed
    pkg_files = sorted({f for f in files
                        if f.startswith(('/usr/', '/opt/', '/lib/', '/lib64/',
                                         '/bin/', '/sbin/', '/etc/'))})
    if not pkg_files:
        return set()

    packages = set()
    batch_size = 100
    for i in range(0, len(pkg_files), batch_size):
        batch = pkg_files[i:i + batch_size]
        try:
            result = subprocess.run(
                ["dpkg", "-S"] + batch,
                capture_output=True, text=True, timeout=30,
            )
            # Parse output even if returncode != 0 (some files may not be in packages)
            for line in result.stdout.splitlines():
                if ': ' in line:
                    pkg_part = line.split(': ')[0]
                    # Handle "pkg1, pkg2: /path" format and :arch suffixes
                    for pkg in pkg_part.split(','):
                        pkg = pkg.strip().split(':')[0]
                        if pkg:
                            packages.add(pkg)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return packages


def find_nmrbox_requirements(open_files, nmrbox_pkgs, dep_to_nmrbox):
    """Match open files against installed NMRBox packages.

    Returns a list of (software, version) tuples for NMRBox packages whose files
    (or whose dependencies' files) were opened by the monitored process.
    """
    # Also resolve symlinks so we match regardless of path form
    resolved = set()
    for f in open_files:
        try:
            resolved.add(os.path.realpath(f))
        except (OSError, ValueError):
            pass
    all_paths = open_files | resolved

    owning_packages = _find_owning_packages(all_paths)

    found = {}
    for pkg in owning_packages:
        # Direct match: the package itself has NMRBox metadata
        if pkg in nmrbox_pkgs:
            found[nmrbox_pkgs[pkg]] = True
        # Reverse dependency: this package is a dep of an NMRBox package
        if pkg in dep_to_nmrbox:
            for sw_ver in dep_to_nmrbox[pkg]:
                found[sw_ver] = True

    return sorted(found.keys())


CONDOR_MAPPINGS = (('-', ''),
                   ('+', 'PLUS'),
                   (' ', '_'),
                   )


def valid_condor_identifier(identifier):
    """Replace invalid CONDOR characters.
    Logic coordinated with svn+ssh://devel.nmrbox.org/svn/nmrbox/trunk/python-script/services/productionmonitor/src/productionmonitor/main.py"""
    if str.isdigit(identifier[0]):
        identifier = 'v' + identifier
    for old, new in CONDOR_MAPPINGS:
        identifier = identifier.replace(old, new)
    return identifier


def format_nmrbox_requirement(software, version):
    """Format NMRBox software/version into a condor requirement string."""
    formatted_version = valid_condor_identifier(version)
    return f'{software} == "{formatted_version}"'


def format_memory_mb(kb):
    """Convert KB to MB with 25% headroom, rounded up to nearest 64 MB."""
    mb = kb / 1024
    mb_with_headroom = mb * 1.25
    # Round up to nearest 64 MB for cleaner values
    return max(64, int(math.ceil(mb_with_headroom / 64) * 64))


def validate_requirements(nmrbox_req_str):
    """Check condor_status to see if any nodes match the given requirement.
    Returns (matches: int or None) - None if condor_status is unavailable."""
    if not nmrbox_req_str:
        return None
    try:
        result = subprocess.run(
            ["condor_status", "-const", nmrbox_req_str, "-total"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        # Count lines that look like totals with machine counts
        # The -total output ends with a summary line like "Total   N ..."
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if parts and parts[0] == "Total":
                try:
                    return int(parts[1])
                except (IndexError, ValueError):
                    pass
        # If we got output but no Total line, count non-header non-empty lines
        return 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def interactive_confirm(peak_rss_kb, avg_cpus, gpu_used, gpu_memory_mb,
                        nmrbox_packages, still_climbing):
    """Present findings to user and let them adjust before writing submit file.
    Returns (memory_mb, cpus, use_gpu, gpu_mem_mb, nmrbox_req_strs)."""
    suggested_mem = format_memory_mb(peak_rss_kb)
    peak_mb = peak_rss_kb / 1024

    # Cap suggested CPUs at 32; jobs requesting more take much longer to match.
    max_cpus = 32
    suggested_cpus = min(avg_cpus, max_cpus)

    print("\n" + "=" * 60)
    print("  Condorize - Detected Settings")
    print("=" * 60)
    print(f"  Peak memory (RSS):  {peak_mb:.1f} MB")
    print(f"  Suggested request:  {suggested_mem} MB (with 25% headroom)")
    if avg_cpus > max_cpus:
        print(f"  Avg CPUs used:      {avg_cpus} (capped to {max_cpus})")
    else:
        print(f"  Avg CPUs used:      {avg_cpus}")
    print(f"  GPU used:           {'Yes' if gpu_used else 'No'}"
          f"{f' ({gpu_memory_mb} MB)' if gpu_memory_mb else ''}")
    if nmrbox_packages:
        for sw, ver in nmrbox_packages:
            req = format_nmrbox_requirement(sw, ver)
            print(f"  NMRBox requirement: {req}")
    else:
        print(f"  NMRBox requirement: None")
    print("=" * 60)

    if still_climbing:
        print()
        print("  WARNING: Memory usage was still increasing when monitoring")
        print("  stopped. The actual memory needs of your program may be")
        print("  higher than what was observed. Consider increasing the")
        print("  memory request or running with a longer --timeout.")

    print()
    print("  Review the settings below. Press Enter to accept the")
    print("  suggested value shown in [brackets], or type a new value.")
    print()

    # Memory
    while True:
        ans = input(f"  Memory to request in MB [{suggested_mem}]: ").strip()
        if not ans:
            memory_mb = suggested_mem
            break
        try:
            memory_mb = int(ans)
            if memory_mb <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive integer.")

    # CPUs
    while True:
        ans = input(f"  CPUs to request [{suggested_cpus}]: ").strip()
        if not ans:
            cpus = suggested_cpus
            break
        try:
            cpus = int(ans)
            if cpus <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive integer.")

    if cpus > 32:
        print()
        print(f"  WARNING: Requesting {cpus} CPUs. Jobs requesting more than")
        print(f"  32 CPUs may take much longer to match to a machine.")
        print(f"  If your program does not actually need this many CPUs,")
        print(f"  consider reducing the request.")
        while True:
            ans = input(f"  Proceed with {cpus} CPUs, or enter a new value [{cpus}]: ").strip()
            if not ans:
                break
            try:
                cpus = int(ans)
                if cpus <= 0:
                    raise ValueError
                break
            except ValueError:
                print("  Please enter a positive integer.")

    # GPU
    while True:
        ans = input(f"  Request a GPU? [{'Y/n' if gpu_used else 'y/N'}]: ").strip().lower()
        if not ans:
            use_gpu = gpu_used
            break
        if ans in ("y", "yes"):
            use_gpu = True
            break
        if ans in ("n", "no"):
            use_gpu = False
            break
        print("  Please enter y or n.")

    gpu_mem_mb = 0
    if use_gpu and gpu_memory_mb > 0:
        suggested_gpu = int(math.ceil(gpu_memory_mb * 1.25 / 64) * 64)
        ans = input(f"  GPU memory to request in MB [{suggested_gpu}]: ").strip()
        if ans:
            try:
                gpu_mem_mb = int(ans)
            except ValueError:
                gpu_mem_mb = suggested_gpu
        else:
            gpu_mem_mb = suggested_gpu

    # NMRBox requirements
    nmrbox_req_strs = []
    if nmrbox_packages:
        for sw, ver in nmrbox_packages:
            req = format_nmrbox_requirement(sw, ver)
            ans = input(f"  Include NMRBox requirement '{req}'? [Y/n]: ").strip().lower()
            if ans not in ("n", "no"):
                nmrbox_req_strs.append(req)
        # Validate combined requirements against the pool
        if nmrbox_req_strs:
            combined = ' && '.join(nmrbox_req_strs)
            matches = validate_requirements(combined)
            if matches is not None and matches == 0:
                print()
                print(f"  WARNING: No machines in the pool currently match the")
                print(f"  requirements '{combined}'. Your job may remain")
                print(f"  idle indefinitely. You may want to check the available")
                print(f"  versions with:")
                for sw, _ver in nmrbox_packages:
                    print(f"    condor_status -af {sw}")

    return memory_mb, cpus, use_gpu, gpu_mem_mb, nmrbox_req_strs


def needs_file_transfer(binary_path, cmd_args):
    """Check if any relevant path is on an ephemeral filesystem (/scratch or /tmp).
    Checks the executable path, cwd, and all command arguments that look like paths."""
    paths_to_check = [binary_path, os.getcwd()]
    for arg in cmd_args:
        if os.sep in arg or arg.startswith("/"):
            paths_to_check.append(arg)
    for p in paths_to_check:
        resolved = os.path.realpath(p) if p else ""
        if resolved == "/scratch" or resolved.startswith("/scratch/"):
            return True
        if resolved == "/tmp" or resolved.startswith("/tmp/"):
            return True
    return False


def condor_quote_args(args):
    """Quote arguments for HTCondor new-style syntax.
    Returns a double-quoted string where args with special characters
    are individually single-quoted, and literal single quotes are doubled."""
    # Characters that require an argument to be single-quoted
    special = set(" \t\n\"'\\")
    quoted_parts = []
    for arg in args:
        if not arg:
            # Empty argument needs quoting
            quoted_parts.append("''")
        elif any(c in special for c in arg):
            # Escape single quotes by doubling them, then wrap in single quotes
            escaped = arg.replace("'", "''")
            quoted_parts.append(f"'{escaped}'")
        else:
            quoted_parts.append(arg)
    return '"' + " ".join(quoted_parts) + '"'


def resolve_submit_filename(output_path, binary_path):
    """Determine the submit filename, checking for collisions.
    Returns the final filename to use."""
    if output_path:
        submit_filename = output_path
    else:
        cmd_name = os.path.basename(binary_path)
        submit_filename = f"{cmd_name}.sub"

    if os.path.exists(submit_filename):
        while True:
            ans = input(f"\n  '{submit_filename}' already exists. Overwrite? [y/N]: ").strip().lower()
            if not ans or ans in ("n", "no"):
                while True:
                    new_name = input("  Enter a new filename: ").strip()
                    if new_name:
                        submit_filename = new_name
                        if os.path.exists(submit_filename):
                            ans2 = input(f"  '{submit_filename}' also exists. Overwrite? [y/N]: ").strip().lower()
                            if ans2 in ("y", "yes"):
                                break
                        else:
                            break
                break
            if ans in ("y", "yes"):
                break

    return submit_filename


def write_submit_file(submit_filename, binary_path, cmd_args, memory_mb, cpus,
                      use_gpu, gpu_mem_mb, nmrbox_req_strs):
    """Write the HTCondor submit file."""
    cmd_name = os.path.basename(binary_path)

    # Build arguments string using HTCondor new-style quoting.
    # The whole value is wrapped in double quotes. Individual arguments
    # that contain spaces, quotes, or special characters are wrapped in
    # single quotes, with literal single quotes escaped as ''.
    arguments = condor_quote_args(cmd_args) if cmd_args else ""
    transfer_files = needs_file_transfer(binary_path, cmd_args)

    # Build requirements list
    requirements = list(nmrbox_req_strs)

    lines = []
    lines.append(f"universe = vanilla")
    lines.append(f"executable = {binary_path}")
    if arguments:
        lines.append(f"arguments = {arguments}")
    lines.append(f"initialdir = {os.getcwd()}")
    lines.append(f"")
    lines.append(f"request_memory = {memory_mb}")
    lines.append(f"request_cpus = {cpus}")
    lines.append(f"request_disk = 2GB")
    if use_gpu:
        lines.append(f"request_gpus = 1")
        if gpu_mem_mb > 0:
            lines.append(f"require_gpus = (GlobalMemoryMb >= {gpu_mem_mb})")
    lines.append(f"")
    if requirements:
        lines.append(f"requirements = {' && '.join(requirements)}")
        lines.append(f"")
    lines.append(f"+Production = True")
    lines.append(f"")
    lines.append(f"output = {cmd_name}.$(Cluster).$(Process).out")
    lines.append(f"error = {cmd_name}.$(Cluster).$(Process).err")
    lines.append(f"log = {cmd_name}.$(Cluster).$(Process).log")
    lines.append(f"")
    if transfer_files:
        lines.append(f"should_transfer_files = IF_NEEDED")
    else:
        lines.append(f"should_transfer_files = NO")
    lines.append(f"getenv = True")
    lines.append(f"queue")
    lines.append(f"")

    content = "\n".join(lines)

    with open(submit_filename, "w") as f:
        f.write(content)

    print(f"\nSubmit file written to: {submit_filename}")
    print(f"Submit with: condor_submit {submit_filename}")
    print(f"\nContents:")
    print("-" * 40)
    print(content)
    return submit_filename


def main():
    timeout, output_path, cmd = parse_args()

    # Resolve binary path
    binary_path = shutil.which(cmd[0]) or cmd[0]

    # Pre-compute NMRBox package data in background while we monitor
    nmrbox_data = {}

    def _precompute():
        pkgs, dep_map = precompute_nmrbox_data()
        nmrbox_data["pkgs"] = pkgs
        nmrbox_data["dep_map"] = dep_map

    precompute_thread = threading.Thread(target=_precompute, daemon=True)
    precompute_thread.start()

    # Monitor the process
    peak_rss_kb, avg_cpus, gpu_used, gpu_memory_mb, still_climbing, exited_early, exit_code, all_open_files = \
        monitor_process(cmd, timeout)

    # Warn if the process failed quickly
    if exited_early and exit_code is not None and exit_code != 0:
        elapsed_msg = "immediately" if peak_rss_kb == 0 else "before the monitoring period ended"
        print(f"\n  WARNING: The process exited {elapsed_msg} with error code {exit_code}.")
        print(f"  The monitored resource usage may not reflect actual needs.")
        ans = input(f"  Continue generating a submit file anyway? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(1)

    # Wait for NMRBox pre-computation to finish (should already be done)
    precompute_thread.join(timeout=15)
    nmrbox_pkgs = nmrbox_data.get("pkgs", {})
    dep_map = nmrbox_data.get("dep_map", {})

    # Find NMRBox packages used by the monitored process
    nmrbox_packages = find_nmrbox_requirements(all_open_files, nmrbox_pkgs, dep_map)

    # Interactive confirmation
    memory_mb, cpus, use_gpu, gpu_mem_mb, nmrbox_req_strs = interactive_confirm(
        peak_rss_kb, avg_cpus, gpu_used, gpu_memory_mb,
        nmrbox_packages, still_climbing
    )

    # Resolve output filename (check for collisions)
    submit_filename = resolve_submit_filename(output_path, binary_path)

    # Write submit file
    cmd_args = cmd[1:] if len(cmd) > 1 else []
    write_submit_file(submit_filename, binary_path, cmd_args, memory_mb, cpus,
                      use_gpu, gpu_mem_mb, nmrbox_req_strs)


if __name__ == "__main__":
    main()
