"""
Quick diagnostic to verify Supabase URL and credentials.

This script helps you:
1. Check DNS resolution of your Supabase URL
2. Test HTTP connectivity
3. Verify API key format
4. Confirm the Supabase service is reachable

Usage:
    python verify_supabase_url.py
"""

import os
import socket
import re
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv


def check_url_format(url: str) -> bool:
    """Check if URL has correct Supabase format."""
    pattern = r"https://[a-z]{20}\.supabase\.co"
    if re.match(pattern, url):
        print("✅ URL format is valid")
        return True
    else:
        print("❌ URL format is INVALID")
        print(f"   Expected format: https://xxxxxxxxxxxxxxxxxxxx.supabase.co")
        print(f"   Your URL: {url}")
        return False


def check_dns(hostname: str) -> bool:
    """Check DNS resolution."""
    print(f"\n🔍 Checking DNS resolution for: {hostname}")
    try:
        info = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
        ips = {addr[4][0] for addr in info}
        print(f"✅ DNS resolves to: {', '.join(ips)}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS resolution FAILED: {e}")
        print("\n⚠️  This usually means:")
        print("   1. The project URL is incorrect")
        print("   2. The project was deleted")
        print("   3. There's a typo in the URL")
        return False


def check_http_connectivity(url: str, api_key: str) -> bool:
    """Test HTTP connectivity to Supabase."""
    print(f"\n🌐 Testing HTTP connectivity...")

    # Try the health endpoint
    health_url = f"{url}/auth/v1/health"

    try:
        response = requests.get(
            health_url,
            headers={"apikey": api_key},
            timeout=10
        )

        if response.status_code == 200:
            print(f"✅ HTTP connection successful (status: {response.status_code})")
            return True
        else:
            print(f"⚠️  HTTP returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP connection FAILED: {e}")
        return False


def check_api_key_format(key: str) -> bool:
    """Check if API key looks valid."""
    if not key:
        print("❌ API key is empty")
        return False

    if len(key) < 30:
        print(f"❌ API key is too short ({len(key)} chars)")
        return False

    # Supabase keys are typically JWT tokens (eyJ...)
    if key.startswith("eyJ"):
        print("✅ API key format looks valid (JWT)")
        return True
    else:
        print("⚠️  API key doesn't look like a JWT token")
        print(f"   Expected to start with 'eyJ', got: {key[:10]}...")
        return False


def main():
    print("="*70)
    print("🔍 SUPABASE CONNECTION DIAGNOSTICS")
    print("="*70)

    # Load environment variables
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    print(f"\n📋 Loaded from .env:")
    print(f"   URL: {url}")
    print(f"   Key: {key[:20]}... (truncated)" if key else "   Key: None")

    if not url or not key:
        print("\n❌ CRITICAL: Missing SUPABASE_URL or SUPABASE_KEY in .env")
        return False

    # 1. Check URL format
    print("\n" + "="*70)
    print("STEP 1: URL Format Check")
    print("="*70)
    url_valid = check_url_format(url)

    # 2. Check API key format
    print("\n" + "="*70)
    print("STEP 2: API Key Format Check")
    print("="*70)
    key_valid = check_api_key_format(key)

    # 3. Check DNS
    print("\n" + "="*70)
    print("STEP 3: DNS Resolution Check")
    print("="*70)
    parsed = urlparse(url)
    hostname = parsed.hostname
    dns_valid = check_dns(hostname)

    if not dns_valid:
        print("\n❌ Cannot proceed - DNS resolution failed")
        print("\n📝 ACTION REQUIRED:")
        print("   1. Log into https://app.supabase.com")
        print("   2. Find your project (or create a new one)")
        print("   3. Go to Settings > API")
        print("   4. Copy the exact 'Project URL' shown")
        print("   5. Update your .env file with the correct URL")
        return False

    # 4. Check HTTP connectivity
    print("\n" + "="*70)
    print("STEP 4: HTTP Connectivity Check")
    print("="*70)
    http_valid = check_http_connectivity(url, key)

    # Final summary
    print("\n" + "="*70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*70)
    print(f"URL Format:        {'✅ PASS' if url_valid else '❌ FAIL'}")
    print(f"API Key Format:    {'✅ PASS' if key_valid else '❌ FAIL'}")
    print(f"DNS Resolution:    {'✅ PASS' if dns_valid else '❌ FAIL'}")
    print(f"HTTP Connectivity: {'✅ PASS' if http_valid else '❌ FAIL'}")

    all_pass = url_valid and key_valid and dns_valid and http_valid

    if all_pass:
        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED - Supabase connection is ready!")
        print("="*70)
        print("\nNext step: Run test_supabase_connection.py")
        return True
    else:
        print("\n" + "="*70)
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print("="*70)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
