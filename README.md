# ParkBru showcase

Local-only React + TypeScript app built with Vite, Tailwind CSS and Lucide icons.

## Run locally

```sh
npm install
npm run dev -- --port 5174
```

## Verify

```sh
npm test
npm run build
```

## Showcase flow

The app starts unparked on each reload and reset. Parking bays and sectors are informational; the app has no reservation or manual session-start flow. Use the profile avatar, then **Simulate sensor arrival** to demonstrate an IoT event detecting the vehicle at The Mall Gadong, Level 2, Bay B-08. The timer starts automatically. **Pay BND {amount} via tarus** demonstrates direct payment without a wallet. Payment leaves the vehicle parked; **Simulate sensor departure** closes the session. **Reset showcase** clears parking, receipts, verification and alerts.

Dashboard shows Occupancy Forecast, shaded/sunlit sectors and concept-style filters. A parking-session card appears only while parking is detected. Navigate changes the displayed map area without changing a detected vehicle location. Bottom navigation: Dashboard, Navigate, My Car, Pay, Alerts.

All IoT events, forecasts, occupancy, Tarus payments, verification and safety actions are simulated. No device backend, payment service, messaging or emergency service is connected. PWA installation and offline shell caching are available in production builds on localhost or HTTPS. Do not deploy, host or upload further changes without an explicit user request.
