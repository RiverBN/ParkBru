# ParkBru showcase

An interactive React + TypeScript showcase, built with Vite, Tailwind CSS and Lucide icons. Responsive desktop and mobile layouts, installable PWA, cached offline demonstration, and local demo wallet/session persistence.

## Run

```sh
npm install
npm run dev
```

## Production

```sh
npm run build
npm run preview
```

Deploy the `dist` directory to an HTTPS static host. PWA installation works on localhost or HTTPS in supported browsers. Offline support becomes available after visiting the built app online. Development mode does not register the service worker.

## Showcase flows

- Overview: occupancy forecast, session timer, wallet, parking map and priority access.
- Find parking: search locations, filter space types and select available bays.
- My parking: elapsed timer, car location and session payment.
- Wallet & passes: demo top-ups, hourly/daily pass selection and activity.
- Safety centre: simulated blocked-driver notification, obstacle reports and SOS.
- Profile: reset the showcase for a new presentation.

All occupancy, forecasts, routes, transactions, verification and safety actions are simulated. No real payment provider, parking operator, messaging or emergency service is connected. The displayed vehicle, account and receipts are demo fixtures. Wallet and active-session flags persist locally; the demo timer and activity are reset on reload.

The ParkBru wordmark is a custom placeholder pending the supplied brand asset. Fonts use Google Fonts with local system fallbacks. Native App Store / Play Store packaging with Capacitor is a future step.
