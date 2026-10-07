"""Quick test script for all API endpoints."""
import urllib.request, urllib.error, json, sys
from PIL import Image
import io

BASE = "http://localhost:8000"

# ── Health ──
print("=== Health ===")
try:
    r = urllib.request.urlopen(f"{BASE}/api/health", timeout=5)
    print(json.dumps(json.loads(r.read()), indent=2))
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)

# ── Image ──
print("\n=== Image Endpoint ===")
try:
    import requests  # use requests for easy multipart
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = requests.post(f"{BASE}/api/detect/image", files={"file": ("test.jpg", buf, "image/jpeg")}, timeout=60)
    print(r.status_code, json.dumps(r.json(), indent=2))
except Exception as e:
    print("FAILED:", e)

print("\nDone.")
