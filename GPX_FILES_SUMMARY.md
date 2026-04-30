# GPX Route Files Summary

## 📁 Available Route Files

### 1. **colombo_city_route.gpx** ⭐ RECOMMENDED FOR QUICK TESTING
- **Type**: Urban Loop
- **Start**: Colombo Fort
- **End**: Colombo Fort (circular)
- **Distance**: ~20 km
- **Duration**: 26 minutes
- **Waypoints**: 15
- **Terrain**: City streets, coastal road
- **Best For**: Quick testing, demo purposes

**Route Path**:
```
Colombo Fort → Pettah → Galle Face → Kollupitiya → 
Bambalapitiya → Wellawatte → Dehiwala → Mount Lavinia → 
Ratmalana → Nugegoda → Borella → Back to Fort
```

---

### 2. **colombo_to_kandy_route.gpx**
- **Type**: Highway Journey
- **Start**: Colombo Fort
- **End**: Kandy City
- **Distance**: ~115 km
- **Duration**: 100 minutes
- **Waypoints**: 25
- **Terrain**: Urban → Highway → Hill Country
- **Best For**: Long-distance testing, elevation changes

**Route Path**:
```
Colombo → Kelaniya → Kadawatha → Gampaha → Nittambuwa → 
Warakapola → Kegalle → Mawanella → Kadugannawa → 
Peradeniya → Kandy
```

**Elevation**: 5m → 465m (climbing to hill country)

---

### 3. **colombo_to_galle_coastal_route.gpx**
- **Type**: Coastal Highway
- **Start**: Colombo Fort
- **End**: Galle Fort
- **Distance**: ~120 km
- **Duration**: 80 minutes
- **Waypoints**: 16
- **Terrain**: Coastal expressway, beach towns
- **Best For**: Scenic route testing, expressway simulation

**Route Path**:
```
Colombo → Mount Lavinia → Moratuwa → Panadura → Kalutara → 
Beruwala → Bentota → Hikkaduwa → Galle Fort
```

**Highlights**: Famous beach towns, Southern Expressway

---

## 🎯 Which Route to Use?

### For Quick Testing (5-10 minutes):
✅ **Use**: `colombo_city_route.gpx` at **4x speed**
- Fast verification of location tracking
- Multiple address changes
- Urban environment

### For Demo/Presentation:
✅ **Use**: `colombo_city_route.gpx` at **2x speed**
- Shows realistic city driving
- Recognizable Sri Lankan locations
- Good for screenshots

### For Comprehensive Testing:
✅ **Use**: `colombo_to_kandy_route.gpx` at **2x speed**
- Tests long-duration monitoring
- Elevation changes
- Highway conditions

### For Scenic/Tourist Route:
✅ **Use**: `colombo_to_galle_coastal_route.gpx` at **2x speed**
- Beautiful coastal route
- Famous beach destinations
- Expressway simulation

---

## 📊 Route Comparison Table

| Route | Distance | Time | Speed | Waypoints | Terrain | Use Case |
|-------|----------|------|-------|-----------|---------|----------|
| **City Loop** | 20 km | 26 min | Urban | 15 | Flat | Quick test ⭐ |
| **To Kandy** | 115 km | 100 min | Highway | 25 | Hills | Long test |
| **To Galle** | 120 km | 80 min | Expressway | 16 | Coastal | Scenic test |

---

## 🚀 Quick Start Commands

### Load City Route (Fastest):
1. Open emulator Extended Controls (`Ctrl + Shift + P`)
2. Click "Location" → "Load GPX/KML"
3. Select `colombo_city_route.gpx`
4. Click Play ▶ at 2x speed

### Set Single Location (Instant):
```bash
# Colombo
adb emu geo fix 79.8612 6.9271

# Kandy
adb emu geo fix 80.6337 7.2906

# Galle
adb emu geo fix 80.2210 6.0535
```

---

## 📍 Key Locations in Routes

### Colombo Area:
- **Fort**: 6.9271, 79.8612 (Starting point)
- **Galle Face**: 6.9319, 79.8478 (Landmark)
- **Mount Lavinia**: 6.8389, 79.8634 (Beach)

### Hill Country:
- **Kandy**: 7.2906, 80.6337 (Cultural capital)
- **Peradeniya**: 7.2567, 80.5678 (Botanical gardens)
- **Kadugannawa**: 7.4012, 80.4012 (Mountain pass)

### Southern Coast:
- **Galle Fort**: 6.0535, 80.2210 (UNESCO site)
- **Hikkaduwa**: 6.1234, 80.1456 (Beach resort)
- **Bentota**: 6.4567, 80.0123 (Water sports)

---

## 🎮 Playback Tips

### Speed Recommendations:
- **0.5x**: Very slow, detailed testing
- **1x**: Real-time simulation
- **2x**: ⭐ Recommended for most testing
- **4x**: Quick verification
- **8x**: Very fast, may skip waypoints

### Testing Scenarios:

#### Scenario 1: Location Accuracy Test
- **Route**: City Loop
- **Speed**: 2x
- **Focus**: Verify coordinates and addresses update correctly

#### Scenario 2: Emergency Alert Test
- **Route**: Any route
- **Speed**: 1x
- **Action**: Simulate drowsiness at specific locations
- **Verify**: Emergency contacts receive correct location

#### Scenario 3: Long-Duration Monitoring
- **Route**: Colombo to Kandy
- **Speed**: 2x
- **Focus**: Test system stability over extended period

#### Scenario 4: Address Resolution Test
- **Route**: Coastal route
- **Speed**: 1x
- **Focus**: Verify geocoding works for various locations

---

## 🔧 Customization

### Modify Existing Routes:
1. Open GPX file in text editor
2. Edit `<trkpt>` elements:
   ```xml
   <trkpt lat="YOUR_LAT" lon="YOUR_LON">
     <ele>ELEVATION</ele>
     <time>TIMESTAMP</time>
     <name>Location Name</name>
   </trkpt>
   ```
3. Save and reload in emulator

### Add New Waypoints:
- Insert new `<trkpt>` between existing points
- Maintain chronological time order
- Use realistic spacing (100-500m apart)

### Create New Routes:
- Use online tools: GPSies, Ride with GPS
- Export as GPX format
- Load in emulator

---

## 📱 Expected App Behavior

### During Route Playback:

```
┌─────────────────────────────────┐
│ 📍 Current Location             │
│ Lat: 6.927100, Lon: 79.861200   │
│ Colombo Fort, Colombo,          │
│ Western Province, Sri Lanka     │
│ Updated: 3s ago                 │
└─────────────────────────────────┘
        ↓ (after 2 minutes)
┌─────────────────────────────────┐
│ 📍 Current Location             │
│ Lat: 6.931900, Lon: 79.847800   │
│ Galle Face Green, Colombo,      │
│ Western Province, Sri Lanka     │
│ Updated: 2s ago                 │
└─────────────────────────────────┘
```

### Update Frequency:
- **Coordinates**: Every 10 meters of movement
- **Address**: 1-2 seconds after coordinate change
- **Timestamp**: Real-time

---

## ⚠️ Important Notes

1. **Emulator Performance**: 
   - Allocate sufficient RAM (2GB+)
   - Enable hardware acceleration
   - Close unnecessary apps

2. **Location Permission**:
   - Must be granted in app
   - Check Android settings if not working

3. **Geocoding Limits**:
   - May be rate-limited
   - Some addresses may not resolve
   - Normal behavior for remote areas

4. **Route Looping**:
   - Routes can be played repeatedly
   - Use for continuous testing
   - Reset to start with progress bar

---

## 📚 Additional Resources

- **Full Guide**: See `GPX_ROUTE_SIMULATION_GUIDE.md`
- **Quick Setup**: See `QUICK_GPX_SETUP.md`
- **Location Service**: See `mobile_app/lib/services/location_service.dart`
- **Implementation**: See `mobile_app/LOCATION_TRACKING_IMPLEMENTATION.md`

---

## 🎯 Success Checklist

✅ GPX file loaded in emulator  
✅ Route visible on map  
✅ Playback controls working  
✅ App showing location updates  
✅ Coordinates changing along route  
✅ Addresses resolving correctly  
✅ Timestamp updating  
✅ No errors in console  

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Location not updating | Grant location permission |
| Address unavailable | Normal, geocoding may be slow |
| Route too fast | Reduce speed to 1x or 2x |
| App crashes | Check memory allocation |
| Wrong country shown | Reload GPX file, restart emulator |

---

## 📞 Support

For issues or questions:
1. Check `GPX_ROUTE_SIMULATION_GUIDE.md` for detailed help
2. Verify emulator settings
3. Test with single location first (`adb emu geo fix`)
4. Check app logs for errors

---

**Happy Testing! 🚗💤📍**
