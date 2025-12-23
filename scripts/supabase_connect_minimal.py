"""Minimal Supabase connectivity probe.

Checks that env vars are present, resolves DNS, and performs a simple HTTPS
request to the Supabase health endpoint to confirm reachability without using
the Supabase SDK.
"""

import os
import socket
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    print("SUPABASE_URL:", url)
    print("SUPABASE_KEY (prefix):", key[:12] + "..." if key else None)

    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_KEY")
        return 1

    parsed = urlparse(url)
    host = parsed.hostname
    print("Parsed host:", host)

    print("\nDNS lookup:")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        unique_ips = {info[4][0] for info in infos}
        print("Resolved IPs:", ", ".join(unique_ips))
    except Exception as exc:
        print("DNS resolution failed:", exc)
        return 1

    print("\nHTTP health check:")
    health_url = f"{parsed.scheme}://{parsed.hostname}/auth/v1/health"
    try:
        resp = requests.get(health_url, timeout=10, headers={"apikey": key})
        print("Status:", resp.status_code)
        print("Body:", resp.text[:200])
    except Exception as exc:
        print("HTTP request failed:", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
