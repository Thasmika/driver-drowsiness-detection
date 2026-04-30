# GPX Setup Visual Guide

## 🎯 Step-by-Step Visual Instructions

### Step 1: Open Android Emulator Extended Controls

```
┌─────────────────────────────────────┐
│  Android Emulator                   │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │      Your App Running         │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  Toolbar (Right Side):              │
│  ┌─┐                                │
│  │⚙│ ← Settings                     │
│  ├─┤                                │
│  │📷│ ← Camera                       │
│  ├─┤                                │
│  │...│ ← Click HERE! (Extended     │
│  └─┘    Controls)                   │
└─────────────────────────────────────┘

OR Press: Ctrl + Shift + P (Windows/Linux)
          Cmd + Shift + P (Mac)
```

---

### Step 2: Navigate to Location Settings

```
┌──────────────────────────────────────────────────┐
│ Extended Controls                          [X]   │
├──────────────┬───────────────────────────────────┤
│              │                                   │
│ Battery      │                                   │
│ Phone        │                                   │
│ Directional  │                                   │
│ Microphone   │                                   │
│ Fingerprint  │                                   │
│ Virtual      │                                   │
│   sensors    │                                   │
│ Bug report   │                                   │
│ Settings     │                                   │
│ Help         │                                   │
│ ► Location   │ ← CLICK HERE!                     │
│ Cell         │                                   │
│ Google Play  │                                   │
│ Snapshots    │                                   │
│ Screen       │                                   │
│   record     │                                   │
│              │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### Step 3: Load GPX File

```
┌──────────────────────────────────────────────────┐
│ Extended Controls - Location            [X]      │
├──────────────┬───────────────────────────────────┤
│              │  ┌─────────────────────────────┐  │
│ Location     │  │                             │  │
│              │  │      Map View               │  │
│              │  │                             │  │
│              │  │   (Route will appear here)  │  │
│              │  │                             │  │
│              │  └─────────────────────────────┘  │
│              │                                   │
│              │  Single points:                   │
│              │  Latitude:  [_____________]       │
│              │  Longitude: [_____________]       │
│              │  [Send]                           │
│              │                                   │
│              │  Routes:                          │
│              │  [Load GPX/KML] ← CLICK HERE!    │
│              │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### Step 4: Select GPX File

```
┌──────────────────────────────────────────────────┐
│ Open File                                  [X]   │
├──────────────────────────────────────────────────┤
│                                                  │
│  📁 drowsiness_detection_app                     │
│    ├─ 📄 colombo_city_route.gpx ← QUICK TEST   │
│    ├─ 📄 colombo_to_kandy_route.gpx             │
│    ├─ 📄 colombo_to_galle_coastal_route.gpx     │
│    ├─ 📁 backend                                 │
│    ├─ 📁 mobile_app                              │
│    └─ ...                                        │
│                                                  │
│  File name: colombo_city_route.gpx              │
│                                                  │
│              [Cancel]  [Open] ← CLICK!           │
└──────────────────────────────────────────────────┘
```

---

### Step 5: Route Loaded Successfully

```
┌──────────────────────────────────────────────────┐
│ Extended Controls - Location            [X]      │
├──────────────┬───────────────────────────────────┤
│              │  ┌─────────────────────────────┐  │
│ Location     │  │  🗺️                         │  │
│              │  │    ●─────●─────●            │  │
│              │  │    │           │            │  │
│              │  │    ●     ●     ●            │  │
│              │  │    │     │     │            │  │
│              │  │    ●─────●─────●            │  │
│              │  │  (Route displayed)          │  │
│              │  └─────────────────────────────┘  │
│              │                                   │
│              │  Route: colombo_city_route.gpx   │
│              │                                   │
│              │  Playback Controls:               │
│              │  ▶ Play  ⏸ Pause  ⏹ Stop        │
│              │                                   │
│              │  Speed: [====|====] 2x            │
│              │                                   │
│              │  Progress: [●─────────────] 0%    │
│              │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### Step 6: Play the Route

```
┌──────────────────────────────────────────────────┐
│ Extended Controls - Location            [X]      │
├──────────────┬───────────────────────────────────┤
│              │  ┌─────────────────────────────┐  │
│ Location     │  │  🗺️                         │  │
│              │  │    ●─────●─────●            │  │
│              │  │    │     🚗    │            │  │
│              │  │    ●     ●     ●            │  │
│              │  │    │     │     │            │  │
│              │  │    ●─────●─────●            │  │
│              │  │  (Car moving along route)   │  │
│              │  └─────────────────────────────┘  │
│              │                                   │
│              │  Route: colombo_city_route.gpx   │
│              │  Status: ▶ PLAYING                │
│              │                                   │
│              │  Playback Controls:               │
│              │  ⏸ Pause  ⏹ Stop                 │
│              │                                   │
│              │  Speed: [====|====] 2x ← ADJUST  │
│              │                                   │
│              │  Progress: [●●●●──────] 35%       │
│              │                                   │
└──────────────┴───────────────────────────────────┘
```

---

### Step 7: Monitor in Your App

```
┌─────────────────────────────────────┐
│  ← Monitoring              ℹ️       │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │   📷 Camera Preview           │  │
│  │   ✅ Face Detected            │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ 📍 Current Location           │  │
│  │ Lat: 6.931900, Lon: 79.847800 │  │ ← UPDATING!
│  │ Galle Face Green, Colombo,    │  │
│  │ Western Province, Sri Lanka   │  │
│  │ Updated: 2s ago               │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │   Normal - Driving Safe       │  │
│  │   Drowsiness: 15%             │  │
│  │   Confidence: 92%             │  │
│  └───────────────────────────────┘  │
│                                     │
│   Face      FPS      Latency       │
│  Detected    1       1000ms        │
│                                     │
└─────────────────────────────────────┘
```

---

## 🎮 Playback Control Details

### Speed Slider:
```
Slower ←─────●─────→ Faster

0.5x   1x   2x   4x   8x
 │     │    │    │    │
 │     │    │    │    └─ Very fast (testing)
 │     │    │    └────── Fast (quick test)
 │     │    └─────────── ⭐ Recommended
 │     └──────────────── Real-time
 └────────────────────── Very slow (detailed)
```

### Progress Bar:
```
[●●●●●●────────────────] 35%
 │                    │
 Start              End

Click anywhere to jump to that point!
```

---

## 📊 What Happens in Real-Time

### Timeline View:

```
Time    Location              App Display
─────────────────────────────────────────────────
00:00   Colombo Fort          Lat: 6.9271, Lon: 79.8612
        (Start)               Colombo Fort, Sri Lanka
                              ↓
00:02   Pettah                Lat: 6.9297, Lon: 79.8567
                              Pettah, Colombo, Sri Lanka
                              ↓
00:04   Galle Face            Lat: 6.9319, Lon: 79.8478
                              Galle Face Green, Colombo
                              ↓
00:06   Galle Face Hotel      Lat: 6.9234, Lon: 79.8456
                              Galle Face, Colombo
                              ↓
00:08   Kollupitiya           Lat: 6.9167, Lon: 79.8512
                              Kollupitiya, Colombo
                              ↓
...     ...                   ...
```

---

## 🔄 Route Comparison Visual

### City Loop (26 minutes):
```
     Colombo Fort
         ●
         │
    Galle Face
         ●
         │
    Kollupitiya
         ●
         │
   Mount Lavinia ──┐
         ●         │
         │         │
    Nugegoda       │
         ●         │
         │         │
    Back to Fort ──┘
         ●
```

### Colombo to Kandy (100 minutes):
```
Colombo ●────────────────────────────────────● Kandy
        │                                    │
        │  Gampaha    Kegalle   Kadugannawa │
        ●─────●─────────●──────────●────────┘
        
Elevation: 5m ────────────────────────→ 465m
                  (Climbing)
```

### Colombo to Galle (80 minutes):
```
Colombo ●
        │
        │  Mount Lavinia
        ●
        │
        │  Kalutara
        ●
        │
        │  Bentota
        ●
        │
        │  Hikkaduwa
        ●
        │
Galle   ●

(Coastal Route 🌊)
```

---

## ✅ Success Indicators

### In Emulator:
```
✅ Route line visible on map
✅ Car icon moving along route
✅ Progress bar advancing
✅ Current location updating
```

### In App:
```
✅ Coordinates changing
✅ Address updating
✅ "X seconds ago" timestamp
✅ No error messages
```

---

## 🎯 Quick Test Checklist

```
□ Emulator running
□ Extended Controls open (Ctrl+Shift+P)
□ Location tab selected
□ GPX file loaded
□ Route visible on map
□ Play button clicked
□ Speed set to 2x
□ App monitoring screen open
□ Location updating in app
□ Address resolving correctly
```

---

## 🆘 Troubleshooting Visual

### Problem: Location Not Updating

```
❌ App shows:
┌─────────────────────────┐
│ 📍 Current Location     │
│ Getting location...     │
│ (Not changing)          │
└─────────────────────────┘

✅ Solution:
1. Check permission:
   Settings → Apps → Drowsiness Detection
   → Permissions → Location → Allow

2. Restart route:
   Extended Controls → Stop → Play

3. Try single location first:
   adb emu geo fix 79.8612 6.9271
```

---

## 📱 Alternative: ADB Commands

### Quick Location Set (No GPX needed):

```
PowerShell/Terminal:
┌────────────────────────────────────────┐
│ PS> adb emu geo fix 79.8612 6.9271    │
│                                        │
│ Setting location to Colombo Fort...   │
│ ✅ Done!                               │
└────────────────────────────────────────┘

App immediately shows:
┌─────────────────────────────────────┐
│ 📍 Current Location                 │
│ Lat: 6.927100, Lon: 79.861200       │
│ Colombo Fort, Colombo, Sri Lanka    │
│ Updated: 1s ago                     │
└─────────────────────────────────────┘
```

---

## 🎬 Complete Workflow Diagram

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Open Emulator       │
│ Extended Controls   │
│ (Ctrl+Shift+P)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Click "Location"    │
│ in sidebar          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Click "Load GPX/KML"│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Select GPX file:    │
│ - City (quick)      │
│ - Kandy (long)      │
│ - Galle (coastal)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Route loads on map  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Click Play ▶        │
│ Set speed to 2x     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Open app            │
│ Go to Monitoring    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Watch location      │
│ update in real-time │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│   Success!  │
│     ✅      │
└─────────────┘
```

---

**You're all set! 🚗📍**

The location will now update automatically as the route plays, showing realistic Sri Lankan locations in your app!
