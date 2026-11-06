# ParkBru showcase

Local-only React + TypeScript app built with Vite, Tailwind CSS and Lucide icons.

## Run locally

```sh
npm install
npm run dev -- --port 5174
```

## Local HTTPS for a phone PWA demo

Connect the laptop and phone to the same hotspot. Install [mkcert](https://github.com/FiloSottile/mkcert) on Windows, then from PowerShell run `./run-showcase-https.ps1`. The script asks which laptop IPv4 address your phone can reach, creates the locally trusted certificate, builds the app, opens the parking detector in a separate window, and starts the HTTPS preview. Open the printed `https://LAPTOP_IP:4173` address on your phone. The detector uses Python 3.13; choose the camera in its window and keep it open while showing the app. Timestamped setup, build, detector, and preview logs are saved under `logs/`.

You can also pass the laptop's hotspot IPv4 address directly:

```powershell
./run-showcase-https.ps1 -Address 192.168.43.20
```

Install mkcert's `rootCA.pem` on the phone: run `mkcert -CAROOT` to find it and transfer only `rootCA.pem`. Never transfer `rootCA-key.pem`. On iPhone, install the certificate profile and enable full trust under **Settings → General → About → Certificate Trust Settings**. On Android, install it from **Security & privacy → Encryption & credentials → Install a certificate**; menu names vary by device.

The script handles the build and server startup. To do these steps manually instead:

```powershell
npm run build
npm run preview:https
```

On the phone, open `https://192.168.43.20:4173` using the same laptop IP. The local HTTPS server proxies `/api` to the laptop's loopback detector bridge, so the detector feed continues to work. If the hotspot assigns a different laptop IP later, generate a new leaf certificate for that IP. The generated `certs` files are ignored by Git.

## Verify

```sh
npm test
npm run build
```

## Showcase flow

The app starts unparked on each reload and reset. Parking bays and sectors are informational; the app has no reservation or manual session-start flow. Use the profile avatar, then **Simulate sensor arrival** to demonstrate an IoT event detecting the vehicle at Demo Location, Level 2, Bay B-08. The timer starts automatically. **Pay via online payment** demonstrates the online payment flow (demo only). Payment leaves the vehicle parked; **Simulate sensor departure** closes the session. **Reset showcase** clears parking, receipts, verification and alerts.

Dashboard shows Occupancy Forecast, shaded/sunlit sectors and concept-style filters. A parking-session card appears only while parking is detected. Navigate changes the displayed map area without changing a detected vehicle location. Bottom navigation: Dashboard, Navigate, My Car, Pay, Alerts.

Forecasts, payment, verification and safety actions remain simulated. The miniature camera detector can report bay occupancy and OCR results to the app through a loopback API; no camera frames leave this computer. No external payment, messaging or emergency service is connected. PWA installation and offline shell caching are available in production builds on localhost or HTTPS. Do not deploy, host or upload further changes without an explicit user request.

## Miniature camera demo

The optional plate reader in `scripts/phone-parking-detect.py` uses PaddleOCR PP-OCRv5 only (no Tesseract fallback). Install PaddleOCR and PaddlePaddle for plate recognition; model files download on first use. Plates must contain 2–3 letters followed by 1–4 digits (for example, BAA1002 or BG12). After a bay is occupied continuously for five seconds, OCR starts automatically and retries once per minute up to five retries. If no valid plate is recognized after the retries, the bay reports `INVALID PLATE`. The miniature plate must be large and sharply focused; OCR can still misread tiny or handwritten plates.

The detector is split into small Python modules so each part can be read independently: `camera_setup.py` opens the USB/Android source and applies camera settings; `bay_vision.py` finds bay outlines and compares frames to detect occupancy; `plate_reader.py` runs and validates OCR; `detector_ui.py` handles selection windows and drawing helpers; `app_bridge.py` forwards bay state to the app. `phone-parking-detect.py` is the entry point and coordinates these pieces.

### Send detections to the app

The detector starts the `scripts/parking-bridge.py` API automatically on `127.0.0.1:8765` and sends bay occupancy and OCR results to it. The app polls that API once per second. Camera frames remain inside the detector and are never sent to the app. Start the app and detector in separate terminals:

1. Start ParkBru with `npm run dev -- --port 5175` and open the local address Vite prints.
2. Start the detector with `py -3.13 scripts/phone-parking-detect.py`. Choose a USB camera or Android IP camera in the setup dialog. For Android, enter the camera app's video URL, such as `https://PHONE_IP:4443/video/mjpeg`.

Keep the app and detector running. Draw or confirm the bays, make sure they are empty, then press **C** to calibrate. When a bay becomes occupied, the app starts a session and displays the reported bay and any plate text the OCR reads. When the bay empties, the app closes the session. The app's session panel shows whether the detector API is connected and which camera source is active. To run the API separately, use `py -3.13 scripts/parking-bridge.py`; the detector will reuse an already-running default API.

The API exposes `GET /api/state` for the current camera/bay status and accepts detector updates with `POST /api/state`. It binds to loopback only and includes CORS access for local ParkBru development origins.
