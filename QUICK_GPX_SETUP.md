# Quick GPX Setup Guide

## 🚀 Fast Setup (3 Steps)

### Step 1: Open Emulator Location Controls
- Click **three dots (...)** on emulator toolbar
- Or press `Ctrl + Shift + P`

### Step 2: Load GPX File
1. Click **"Location"** in left sidebar
2. Click **"Load GPX/KML"** button
3. Select file:
   - `colombo_to_kandy_route.gpx` (long route, 100 min)
   - `colombo_city_route.gpx` (short route, 26 min)

### Step 3: Play Route
- Click **Play button (▶)**
- Set speed to **2x** (recommended)
- Watch location update in app!

---

## 📍 Quick Location Commands

### Set Single Location (Colombo):
```bash
adb emu geo fix 79.8612 6.9271
```

### Set Single Location (Kandy):
```bash
adb emu geo fix 80.6337 7.2906
```

---

## 🎯 What You'll See in App

✅ **Coordinates update** every few seconds  
✅ **Address changes** to match location  
✅ **"X seconds ago"** timestamp updates  
✅ **Real-time tracking** as route plays  

---

## ⚡ Troubleshooting

**Location not updating?**
→ Grant location permission in app

**Address unavailable?**
→ Normal, geocoding may be slow

**Route too fast?**
→ Adjust speed slider to 1x

---

## 📊 Route Comparison

| Route | Distance | Time | Waypoints | Best For |
|-------|----------|------|-----------|----------|
| **Colombo-Kandy** | 115 km | 100 min | 25 | Highway testing |
| **Colombo City** | 20 km | 26 min | 15 | Quick testing |

---

## 🎮 Playback Controls

- **▶ Play**: Start route
- **⏸ Pause**: Stop at current point
- **Speed Slider**: 1x, 2x, 4x, 8x
- **Progress Bar**: Jump to any point

---

## 💡 Pro Tips

1. Use **2x speed** for efficient testing
2. **Pause** to test specific locations
3. **Loop** routes for continuous testing
4. **Monitor** address changes for accuracy
5. **Test emergency alerts** at different points

---

## 📱 Expected App Behavior

```
┌─────────────────────────────┐
│ 📍 Current Location         │
│ Lat: 6.927100, Lon: 79.861200
│ Colombo Fort, Western       │
│ Province, Sri Lanka         │
│ Updated: 2s ago             │
└─────────────────────────────┘
```

Location updates automatically as route plays!

---

## 🔗 Full Documentation

See `GPX_ROUTE_SIMULATION_GUIDE.md` for:
- Detailed instructions
- Custom route creation
- Advanced troubleshooting
- Testing scenarios
