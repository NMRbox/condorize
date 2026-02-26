#!/usr/bin/env python3
"""condorize - Monitor a program's resource usage and generate an HTCondor submit file."""

import argparse
import math
import os
import signal
import subprocess
import shutil
import sys
import threading
import time


def parse_args():
    """Parse command line arguments, splitting on '--' to separate our flags from the target command."""
    argv = sys.argv[1:]

    # Split on '--' if present
    if "--" in argv:
        idx = argv.index("--")
        our_args = argv[:idx]
        cmd_args = argv[idx + 1:]
    else:
        # First non-flag argument starts the command
        our_args = []
        cmd_args = []
        for i, arg in enumerate(argv):
            if not arg.startswith("-"):
                cmd_args = argv[i:]
                break
            our_args.append(arg)
            # Handle --timeout VALUE (grab next arg too)
            if arg == "--timeout" and i + 1 < len(argv):
                our_args.append(argv[i + 1])
                cmd_args = argv[i + 2:]
                break

    parser = argparse.ArgumentParser(
        prog="condorize",
        description="Monitor a command and generate an HTCondor submit file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Monitoring duration in seconds (default: 60)",
    )
    args = parser.parse_args(our_args)

    if not cmd_args:
        parser.error("No command specified. Usage: condorize [--timeout N] -- command [args...]")

    return args.timeout, cmd_args


def get_child_pids(pid):
    """Get all descendant PIDs of a process by walking /proc."""
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
                    child_pid = int(entry)
                    if ppid == pid or ppid in children:
                        children.add(child_pid)
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                continue
    except FileNotFoundError:
        pass
    return children


def get_process_tree_pids(root_pid):
    """Get root PID plus all descendants."""
    pids = {root_pid}
    # Multiple passes to catch nested children
    for _ in range(5):
        new_pids = set()
        for pid in list(pids):
            new_pids.update(get_child_pids(pid))
        if not new_pids - pids:
            break
        pids.update(new_pids)
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


def monitor_process(cmd, timeout):
    """Run cmd for up to timeout seconds, monitoring memory and GPU usage.
    Returns (peak_rss_kb, gpu_used, gpu_memory_mb, exit_early)."""
    print(f"Starting: {' '.join(cmd)}")
    print(f"Monitoring for up to {timeout} seconds...")

    proc = subprocess.Popen(cmd)
    pid = proc.pid

    peak_rss_kb = 0
    gpu_used = False
    gpu_memory_mb = 0
    start = time.time()

    try:
        while time.time() - start < timeout:
            # Check if process already exited
            ret = proc.poll()
            if ret is not None:
                print(f"Process exited on its own (code {ret}) after {time.time() - start:.1f}s")
                break

            # Collect PIDs in the process tree
            pids = get_process_tree_pids(pid)

            # Memory: sum RSS across process tree
            total_rss = sum(read_proc_memory(p) for p in pids)
            peak_rss_kb = max(peak_rss_kb, total_rss)

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
            sys.stdout.write(
                f"\r  [{elapsed:.0f}s/{timeout}s] Peak RSS: {rss_mb:.1f} MB | "
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

    print()
    exit_early = proc.returncode is not None and proc.poll() is not None
    return peak_rss_kb, gpu_used, gpu_memory_mb


def inspect_package(command):
    """Look up the binary's package and check for NMRBox metadata.
    Returns (binary_path, nmrbox_software, nmrbox_version) or (binary_path, None, None).
    Safe to call from a background thread (no direct print output)."""
    binary_path = shutil.which(command)
    if not binary_path:
        return None, None, None

    # Resolve symlinks to get the real path
    binary_path_resolved = os.path.realpath(binary_path)

    # Find which package owns this binary
    try:
        result = subprocess.run(
            ["dpkg", "-S", binary_path_resolved],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            # Try the unresolved path
            result = subprocess.run(
                ["dpkg", "-S", binary_path],
                capture_output=True, text=True, timeout=10,
            )
        if result.returncode != 0:
            return binary_path, None, None

        # Output format: "package: /path/to/file"
        package = result.stdout.strip().split(":")[0]
        # Handle diversion lines or multi-arch suffixes
        package = package.split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return binary_path, None, None

    # Check for NMRBox metadata
    nmrbox_software = None
    nmrbox_version = None

    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f",
             "${Nmrbox-Software}\\n${Nmrbox-Version}\\n", package],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2 and lines[0] and lines[1]:
                nmrbox_software = lines[0].strip()
                nmrbox_version = lines[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: parse apt show output
        try:
            result = subprocess.run(
                ["apt", "show", package],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("Nmrbox-Software:"):
                        nmrbox_software = line.split(":", 1)[1].strip()
                    elif line.startswith("Nmrbox-Version:"):
                        nmrbox_version = line.split(":", 1)[1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return binary_path, nmrbox_software, nmrbox_version


def format_nmrbox_requirement(software, version):
    """Format NMRBox software/version into a condor requirement string.
    Version '1-21' becomes 'v121'."""
    formatted_version = "v" + version.replace("-", "")
    return f'{software} == "{formatted_version}"'


def format_memory_mb(kb):
    """Convert KB to MB with 25% headroom, rounded up to nearest 64 MB."""
    mb = kb / 1024
    mb_with_headroom = mb * 1.25
    # Round up to nearest 64 MB for cleaner values
    return max(64, int(math.ceil(mb_with_headroom / 64) * 64))


def interactive_confirm(peak_rss_kb, gpu_used, gpu_memory_mb, nmrbox_software, nmrbox_version):
    """Present findings to user and let them adjust before writing submit file.
    Returns (memory_mb, use_gpu, gpu_mem_mb, nmrbox_req_str)."""
    suggested_mem = format_memory_mb(peak_rss_kb)
    peak_mb = peak_rss_kb / 1024

    print("\n" + "=" * 60)
    print("  Condorize - Detected Settings")
    print("=" * 60)
    print(f"  Peak memory (RSS):  {peak_mb:.1f} MB")
    print(f"  Suggested request:  {suggested_mem} MB (with 25% headroom)")
    print(f"  GPU used:           {'Yes' if gpu_used else 'No'}"
          f"{f' ({gpu_memory_mb} MB)' if gpu_memory_mb else ''}")
    if nmrbox_software and nmrbox_version:
        req = format_nmrbox_requirement(nmrbox_software, nmrbox_version)
        print(f"  NMRBox requirement: {req}")
    else:
        print(f"  NMRBox requirement: None")
    print("=" * 60)
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

    # NMRBox requirement
    nmrbox_req_str = None
    if nmrbox_software and nmrbox_version:
        req = format_nmrbox_requirement(nmrbox_software, nmrbox_version)
        ans = input(f"  Include NMRBox requirement '{req}'? [Y/n]: ").strip().lower()
        if ans not in ("n", "no"):
            nmrbox_req_str = req

    return memory_mb, use_gpu, gpu_mem_mb, nmrbox_req_str


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


def write_submit_file(binary_path, cmd_args, memory_mb, use_gpu, gpu_mem_mb, nmrbox_req_str):
    """Write the HTCondor submit file."""
    cmd_name = os.path.basename(binary_path)
    submit_filename = f"{cmd_name}.sub"

    # Build arguments string (everything after the executable)
    arguments = " ".join(cmd_args) if cmd_args else ""
    transfer_files = needs_file_transfer(binary_path, cmd_args)

    # Build requirements list
    requirements = []
    if nmrbox_req_str:
        requirements.append(nmrbox_req_str)

    lines = []
    lines.append(f"universe = vanilla")
    lines.append(f"executable = {binary_path}")
    if arguments:
        lines.append(f"arguments = {arguments}")
    lines.append(f"")
    lines.append(f"request_memory = {memory_mb}")
    lines.append(f"request_cpus = 1")
    if use_gpu:
        lines.append(f"request_gpus = 1")
    lines.append(f"")
    if requirements:
        lines.append(f"requirements = {' && '.join(requirements)}")
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
    timeout, cmd = parse_args()

    # Start package inspection in background while we monitor
    pkg_result = {}

    def _inspect():
        bp, sw, ver = inspect_package(cmd[0])
        pkg_result["binary_path"] = bp
        pkg_result["nmrbox_software"] = sw
        pkg_result["nmrbox_version"] = ver

    inspect_thread = threading.Thread(target=_inspect, daemon=True)
    inspect_thread.start()

    # Monitor the process
    peak_rss_kb, gpu_used, gpu_memory_mb = monitor_process(cmd, timeout)

    # Wait for package inspection to finish (should already be done)
    inspect_thread.join(timeout=15)
    binary_path = pkg_result.get("binary_path")
    nmrbox_software = pkg_result.get("nmrbox_software")
    nmrbox_version = pkg_result.get("nmrbox_version")

    if not binary_path:
        binary_path = cmd[0]  # Use as-is if we couldn't resolve it

    # Interactive confirmation
    memory_mb, use_gpu, gpu_mem_mb, nmrbox_req_str = interactive_confirm(
        peak_rss_kb, gpu_used, gpu_memory_mb, nmrbox_software, nmrbox_version
    )

    # Write submit file
    cmd_args = cmd[1:] if len(cmd) > 1 else []
    write_submit_file(binary_path, cmd_args, memory_mb, use_gpu, gpu_mem_mb, nmrbox_req_str)


if __name__ == "__main__":
    main()
