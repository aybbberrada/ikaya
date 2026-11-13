# Ikaya Gaze

A local web app you open on your iPhone. It uses the front camera to track your eyes and draws
where you're looking on the screen in real time, as a circle, a doodle trail or a heatmap.

Everything runs in the browser. Video never leaves the phone.

## Open it online

Hosted on GitHub Pages: **https://aybbberrada.github.io/ikaya/**. Open it in Safari on the iPhone,
tap **Start camera** and allow access. Every push that changes `public/` redeploys it.

## Run it locally

Requirements: Node 18+ and `openssl` (preinstalled on macOS and most Linux).

```bash
npm start          # or: node server.js
```

The server prints an address like `https://192.168.1.23:8443`.

1. Put the iPhone on the **same Wi-Fi** as the computer.
2. Open that `https://` address in **Safari**.
3. Safari warns about the self-signed certificate. Tap **Show Details**, then **visit this website**.
4. Tap **Start camera** and allow camera access.
5. Follow the 15 calibration dots with your **eyes** (keep your head still).
6. Done. Switch between **Circle / Doodle / Heat** in the toolbar.

Tip: in Safari, use Share → **Add to Home Screen** for a full-screen, app-like view.

## Getting good accuracy

- Use bright, even light on your face; avoid strong backlight.
- Hold the phone at 25–40 cm, in portrait, the same way during calibration and use.
- After calibrating, **tap anywhere while looking at that spot** to add a fine-tuning point.
- If you move the phone or your posture a lot, tap **Calibrate** again.
- The status pill shows the estimated average error, measured by predicting each calibration
  point from the others.

Expect roughly finger-sized accuracy (often 1–2 cm on a phone screen). Webcam eye tracking
can't match dedicated infrared trackers.

## How it works

`public/app.js`:

1. **MediaPipe Face Landmarker** (WASM, loaded from CDN) returns 478 face landmarks
   including iris centers, 52 blendshapes (`eyeLookIn/Out/Up/Down…`) and a head-pose matrix.
2. **Features** per frame: iris position inside each eye's corner-to-corner frame, eyelid
   opening, look blendshapes, head rotation and translation, plus a few quadratic terms.
3. **Calibration** records ~1 s of frames at 15 known points, drops outlier frames, and fits a
   **ridge regression** from features to screen position.
4. A **One-Euro filter** smooths the point (the *Smooth* slider). Blinks are ignored.
5. Calibration is saved in `localStorage`; **Use saved calibration** restores it.

The phone needs internet on first load to fetch the MediaPipe library and model from CDNs.
