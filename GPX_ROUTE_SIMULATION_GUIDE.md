# GPX Route Simulation Guide for Android Emulator

## Overview
This guide explains how to use GPX files to simulate realistic driving routes in Sri Lanka for testing the Driver Drowsiness Detection app.

## Available GPX Files

### 1. `colombo_to_kandy_route.gpx`
- **Route**: Colombo Fort → Kandy City
- **Distance**: ~115 km
- **Duration**: ~100 minutes (simulated)
- **Waypoints**: 25 points
- **Terrain**: Urban → Highway → Hill Country
- **Use Case**: Long-distance highway driving simulation

**Route Highlights**:
- Starts at Colombo Fort
- Follows A1 Highway through Gampaha, Nittambuwa
- Passes through Kegalle, Mawanella
- Climbs through Kadugannawa Pass
- Ends at Kandy City

### 2. `colombo_city_route.gpx`
- **Route**: Colombo City Loop
- **Distance**: ~20 km
- **Duration**: ~26 minutes (simulated)
- **Waypoints**: 15 points
- **Terrain**: Urban streets
- **Use Case**: Quick testing, city driving simulation

**Route Highlights**:
- Starts at Colombo Fort
- Goes through Galle Face, Kollupitiya
- Continues to Bambalapitiya, Wellawatte
- Reaches Mount Lavinia
- Returns via Nugegoda, Borella

## How to Load GPX Files in Android Emulator

### Method 1: Using Android Studio Extended Controls (Recommended)

#### Step 1: Open Extended Controls
1. Make sure your Android emulator is running
2. Click the **three dots (...)** button on the emulator toolbar (right side)
3. Or press `Ctrl + Shift + P` (Windows) or `Cmd + Shift + P` (Mac)

#### Step 2: Navigate to Location Settings
1. In the Extended Controls window, click **"Location"** in the left sidebar
2. You'll see the location control panel

#### Step 3: Load GPX File
1. Click on the **"Load GPX/KML"** button at the bottom
2. Navigate to your project directory
3. Select either:
   - `colombo_to_kandy_route.gpx` (long route)
   - `colombo_city_route.gpx` (short route)
4. Click **"Open"**

#### Step 4: Play the Route
1. After loading, you'll see the route displayed on the map
2. Use the playback controls:
   - **Play button** (▶): Start route simulation
   - **Pause button** (⏸): Pause at current location
   - **Speed slider**: Adjust playback speed (1x, 2x, 4x, etc.)
   - **Progress bar**: Jump to specific points on the route

#### Step 5: Monitor in Your App
1. Open the Drowsiness Detection app
2. Navigate to the Monitoring screen
3. Watch the location update in real-time as the route plays
4. The coordinates and address will change as you "drive" along the route

### Method 2: Using ADB Commands (Alternative)

You can also send individual coordinates via ADB:

```bash
# Colombo Fort
adb emu geo fix 79.8612 6.9271

# Galle Face
adb emu geo fix 79.8478 6.9319

# Kandy
adb emu geo fix 80.6337 7.2906
```

**Note**: Order is `longitude` then `latitude` for ADB commands.

## Playback Speed Recommendations

### For Testing Location Updates:
- **1x speed**: Real-time simulation (slow, realistic)
- **2x speed**: Faster testing (recommended)
- **4x speed**: Quick route verification
- **8x speed**: Very fast testing (may miss some waypoints)

### For Drowsiness Detection Testing:
- Use **1x or 2x speed** to allow enough time for:
  - Camera processing
  - Face detection
  - Drowsiness analysis
  - Location updates

## Expected Behavior in App

When playing a GPX route, you should see:

1. **Location Coordinates Update**:
   - Latitude and longitude change every few seconds
   - Coordinates follow the route waypoints

2. **Address Resolution**:
   - Address updates to match current location
   - Shows street, city, province, country
   - May take 1-2 seconds to resolve after coordinate change

3. **Last Update Timestamp**:
   - Shows "Xs ago" or "Xm ago"
   - Updates each time location changes

## Troubleshooting

### Issue: Location Not Updating
**Solution**:
- Check if location permission is granted in the app
- Verify the GPX file loaded successfully
- Try pressing Play button again
- Restart the emulator and reload the GPX file

### Issue: Address Shows "Address unavailable"
**Solution**:
- This is normal for some coordinates
- Geocoding service may be rate-limited
- Try slowing down playback speed
- Some remote areas may not have address data

### Issue: Route Playback Too Fast
**Solution**:
- Adjust the speed slider to 1x or 0.5x
- Pause and manually step through waypoints
- Use the progress bar to control position

### Issue: App Shows Old Location
**Solution**:
- Close and reopen the monitoring screen
- Check if location service is enabled in emulator
- Verify location permission in app settings

## Creating Custom Routes

You can create your own GPX files for specific routes:

### GPX File Structure:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Your Name">
  <trk>
    <name>Route Name</name>
    <trkseg>
      <trkpt lat="LATITUDE" lon="LONGITUDE">
        <ele>ELEVATION</ele>
        <time>TIMESTAMP</time>
        <name>Waypoint Name</name>
      </trkpt>
      <!-- Add more waypoints -->
    </trkseg>
  </trk>
</gpx>
```

### Online GPX Generators:
- **GPSies**: https://www.gpsies.com/
- **Ride with GPS**: https://ridewithgps.com/
- **Google Maps to GPX**: Use browser extensions

### Tips for Creating Routes:
1. Use realistic spacing between waypoints (100-500m)
2. Include elevation data for hill routes
3. Set appropriate time intervals (1-5 minutes)
4. Name waypoints for easy identification
5. Test with short routes first

## Testing Scenarios

### Scenario 1: Highway Driving
- **File**: `colombo_to_kandy_route.gpx`
- **Speed**: 2x
- **Duration**: ~50 minutes
- **Tests**: Long-distance monitoring, location accuracy

### Scenario 2: City Driving
- **File**: `colombo_city_route.gpx`
- **Speed**: 1x
- **Duration**: ~26 minutes
- **Tests**: Frequent location changes, urban navigation

### Scenario 3: Emergency Alert Testing
1. Load any route
2. Start playback at 1x speed
3. Simulate drowsiness (close eyes)
4. Verify emergency contacts receive location
5. Check if location is accurate in alert

## Performance Considerations

### Emulator Settings:
- **RAM**: Allocate at least 2GB for smooth operation
- **Graphics**: Use Hardware acceleration
- **Location**: Enable "Use host GPU"

### App Settings:
- Location updates every 10 meters (configured in LocationService)
- Geocoding throttled to avoid rate limits
- Background location tracking disabled (only active during monitoring)

## Real Device Testing

For the most accurate testing:

1. **Export GPX to Device**:
   - Use apps like "GPS Test" or "Fake GPS Location"
   - Load the GPX file
   - Enable mock locations in Developer Options

2. **Physical Testing**:
   - Drive the actual route in Sri Lanka
   - Compare app behavior with emulator
   - Verify address accuracy

## Additional Resources

### Sri Lankan Coordinates Reference:
| Location | Latitude | Longitude |
|----------|----------|-----------|
| Colombo Fort | 6.9271 | 79.8612 |
| Galle Face | 6.9319 | 79.8478 |
| Mount Lavinia | 6.8389 | 79.8634 |
| Kandy | 7.2906 | 80.6337 |
| Galle | 6.0535 | 80.2210 |
| Jaffna | 9.6615 | 80.0255 |
| Negombo | 7.2008 | 79.8358 |

### Useful Commands:
```bash
# Check emulator location
adb shell dumpsys location

# Set single location
adb emu geo fix <longitude> <latitude>

# Check app permissions
adb shell dumpsys package com.example.drowsiness_detection | grep permission
```

## Summary

1. **Load GPX file** in emulator Extended Controls
2. **Press Play** to start route simulation
3. **Adjust speed** for testing needs
4. **Monitor app** for location updates
5. **Test features** like emergency alerts with location

The GPX route simulation provides a realistic way to test location tracking without physical driving, making it perfect for development and demonstration purposes.
