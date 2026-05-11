"""
FrameAI - Free Public Sharing
Run with: python share.py

Starts the server + creates a public URL via Cloudflare Tunnel (free, no account).
Your friend can open the link in any browser. No login, no download.
Files are auto-deleted after 24 hours.
"""

import subprocess, sys, os, time, urllib.request, re, threading, socket

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
CF_EXE      = os.path.join(BASE_DIR, "cloudflared.exe")
CF_URL      = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

# ── helpers ───────────────────────────────────────────────────────────────────

def _port_open(port=8000, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def _download_cf():
    if os.path.exists(CF_EXE):
        return True
    print("  Downloading Cloudflare tunnel tool (one-time, ~30 MB)...")
    try:
        def _progress(count, block, total):
            pct = min(100, int(count * block * 100 / total))
            print(f"\r  Progress: {pct}%  ", end="", flush=True)
        urllib.request.urlretrieve(CF_URL, CF_EXE, _progress)
        print("\r  Download complete.      ")
        return True
    except Exception as e:
        print(f"\n  ERROR: Could not download cloudflared: {e}")
        print("  Check your internet connection and try again.")
        return False

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║   FrameAI  ·  Free Public Sharing        ║")
    print("  ║   Files auto-deleted after 24 hours      ║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    # Step 1: Start the backend server
    print("  [1/3]  Starting FrameAI server...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=BACKEND_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _port_open(8000, timeout=20):
        print("\n  ERROR: Server failed to start.")
        print("  Make sure you have run:  pip install -r backend/requirements.txt")
        server.terminate()
        input("\n  Press Enter to exit...")
        sys.exit(1)
    print("  [OK]   Server running at http://127.0.0.1:8000")

    # Step 2: Download cloudflared if needed
    print("\n  [2/3]  Preparing tunnel tool...")
    if not _download_cf():
        server.terminate()
        input("\n  Press Enter to exit...")
        sys.exit(1)

    # Step 3: Start tunnel and capture the public URL
    print("\n  [3/3]  Creating your public link (takes ~15 seconds)...")

    tunnel_proc = subprocess.Popen(
        [CF_EXE, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    found_url = None
    deadline  = time.time() + 60  # 60s max to find URL

    for line in tunnel_proc.stdout:
        match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
        if match:
            found_url = match.group(0)
            break
        if time.time() > deadline:
            break

    if found_url:
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║                                                      ║")
        print("  ║   SHARE THIS LINK WITH YOUR FRIEND:                  ║")
        print("  ║                                                      ║")
        print(f"  ║   {found_url:<50} ║")
        print("  ║                                                      ║")
        print("  ║   ✓  Opens in any browser — no install needed       ║")
        print("  ║   ✓  No login required                               ║")
        print("  ║   ✓  Files auto-deleted after 24 hours              ║")
        print("  ║   ✓  Works while this window stays open             ║")
        print("  ║                                                      ║")
        print("  ║   Press Ctrl+C to stop sharing                      ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print()

        try:
            # Keep running until user stops it
            for line in tunnel_proc.stdout:
                pass
            tunnel_proc.wait()
        except KeyboardInterrupt:
            print("\n\n  Sharing stopped. Goodbye!")
    else:
        print("\n  ERROR: Could not get public URL from Cloudflare.")
        print("  This sometimes happens due to network restrictions.")
        print("  Try again or use a different network.")

    tunnel_proc.terminate()
    server.terminate()
    input("\n  Press Enter to exit...")


if __name__ == "__main__":
    main()
