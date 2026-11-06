"""HTTP bridge that forwards bay state to the local ParkBru web app."""

import importlib.util
import json
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_BRIDGE_URL = "http://127.0.0.1:8765/api/state"

def send_bridge_state(server_url: str, bays: list[dict[str, object]], source: str) -> None:
    body = json.dumps({"bays": bays, "source": source}).encode("utf-8")
    request = urllib.request.Request(
        server_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response.read()

def ensure_local_bridge(server_url: str | None):
    """Start the bundled loopback API if the default endpoint is not running."""
    if server_url != DEFAULT_BRIDGE_URL:
        return None
    try:
        with urllib.request.urlopen(server_url, timeout=0.35) as response:
            state = json.loads(response.read())
            if response.status == 200 and "camera_online" in state:
                return None
    except Exception:
        pass

    bridge_path = Path(__file__).with_name("parking-bridge.py")
    if not bridge_path.is_file():
        print(f"ParkBru API file is missing ({bridge_path}); detector will continue without app updates.")
        return None
    spec = importlib.util.spec_from_file_location("parkbru_local_bridge", bridge_path)
    if spec is None or spec.loader is None:
        print("ParkBru API could not be loaded; detector will continue without app updates.")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (OSError, ImportError) as error:
        print(f"ParkBru API could not be loaded ({error}); detector will continue without app updates.")
        return None
    try:
        server = module.create_server()
    except OSError as error:
        print(f"ParkBru API could not start ({error}); detector will continue without app updates.")
        return None
    threading.Thread(target=server.serve_forever, name="parkbru-local-api", daemon=True).start()
    for _ in range(20):
        try:
            with urllib.request.urlopen(server_url, timeout=0.25) as response:
                state = json.loads(response.read())
                if response.status == 200 and "camera_online" in state:
                    print(f"ParkBru API ready at {server_url}")
                    return server
        except Exception:
            time.sleep(0.05)
    server.shutdown()
    server.server_close()
    print("ParkBru API did not respond; detector will continue without app updates.")
    return None

