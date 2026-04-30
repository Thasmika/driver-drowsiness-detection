# Location Tracking Display Implementation

## Overview
Added real-time GPS location tracking and display to the monitoring screen, showing the driver's current location coordinates and address while driving.

## Changes Made

### 1. New Location Service (`lib/services/location_service.dart`)
Created a dedicated service for handling GPS location tracking:
- **getCurrentPosition()**: Gets current GPS coordinates
- **getPositionStream()**: Provides continuous location updates (every 10 meters)
- **formatCoordinates()**: Formats latitude/longitude for display
- **calculateDistance()**: Calculates distance between two positions
- Handles location permissions automatically
- Includes error handling for location service failures

### 2. Updated Dependencies (`pubspec.yaml`)
Added geocoding package for address resolution:
```yaml
geocoding: ^2.1.1
```

### 3. Android Permissions (`android/app/src/main/AndroidManifest.xml`)
Added location permissions:
```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
```

### 4. iOS Permissions (`ios/Runner/Info.plist`)
Added location usage descriptions:
```xml
<key>NSLocationWhenInUseUsageDescription</key>
<string>This app needs location access to display your current driving location and for emergency services.</string>
<key>NSLocationAlwaysUsageDescription</key>
<string>This app needs location access to display your current driving location and for emergency services.</string>
```

### 5. Monitoring Screen Updates (`lib/screens/monitoring_screen.dart`)
Enhanced the monitoring screen with location tracking:

#### New Imports:
- `geolocator` - For GPS position tracking
- `geocoding` - For converting coordinates to addresses
- `location_service.dart` - Custom location service

#### New State Variables:
- `_currentPosition` - Current GPS position
- `_currentAddress` - Human-readable address
- `_lastLocationUpdate` - Timestamp of last location update

#### New Methods:
- `_startLocationTracking()` - Initializes location tracking
- `_updateLocation()` - Gets current location
- `_updateLocationFromPosition()` - Updates UI with new position
- `_formatAddress()` - Formats placemark into readable address
- `_formatLastUpdate()` - Formats time since last update

#### UI Changes:
Added location display panel between camera preview and status panel:
- **Location Icon**: Blue location pin icon
- **Coordinates**: Displays latitude and longitude (6 decimal places)
- **Address**: Shows street, city, state, and country
- **Last Update**: Shows time since last location update
- **Styling**: Dark background with white text for visibility

## Features

### Real-Time Location Tracking
- Automatically starts when monitoring begins
- Updates every 10 meters of movement
- Displays GPS coordinates with high precision
- Shows human-readable address

### Address Resolution
- Converts GPS coordinates to street addresses
- Displays: Street → City → State → Country
- Handles missing address components gracefully
- Shows "Address unavailable" if geocoding fails

### Permission Handling
- Requests location permission on first use
- Continues to work if location permission denied (shows "Getting location...")
- Handles both Android and iOS permission systems

### Visual Design
- Compact display between camera and status panels
- Dark background for better visibility
- Blue location icon for easy identification
- Monospace font for coordinates
- Timestamp showing data freshness

## Usage

When the monitoring screen is opened:
1. Camera permission is requested (existing)
2. Location permission is requested (new)
3. GPS tracking starts automatically
4. Location updates every 10 meters
5. Address is resolved from coordinates
6. Display updates in real-time

## Testing

To test the location tracking:

### On Physical Device:
1. Run the app on a physical device
2. Grant location permissions when prompted
3. Navigate to the monitoring screen
4. Move around to see location updates

### On Emulator:
1. Use Android Studio's emulator location controls
2. Or use command line: `adb emu geo fix <longitude> <latitude>`
3. Example: `adb emu geo fix -122.084 37.422`

### iOS Simulator:
1. Use Xcode's location simulation
2. Features → Location → Custom Location
3. Enter coordinates to test

## Error Handling

The implementation handles various error scenarios:
- **Location services disabled**: Shows "Getting location..."
- **Permission denied**: Continues without location display
- **Geocoding failure**: Shows "Address unavailable"
- **Network issues**: Falls back to coordinates only
- **Stream errors**: Logs to console, continues operation

## Performance Considerations

- Location updates throttled to 10-meter intervals
- Address resolution cached until position changes significantly
- Minimal impact on battery life
- Does not interfere with camera processing

## Integration with Emergency System

The location tracking integrates with the existing emergency contact system:
- Location data available for emergency alerts
- Can be sent to emergency contacts if drowsiness detected
- Provides context for emergency responders

## Future Enhancements

Potential improvements:
1. **Route Tracking**: Record driving route for analysis
2. **Geofencing**: Alert when entering/leaving specific areas
3. **Speed Detection**: Show current driving speed
4. **Location History**: Store location data for trip reports
5. **Offline Maps**: Cache map data for offline use
6. **Nearby Services**: Show nearby rest stops or hospitals

## Dependencies

- `geolocator: ^10.1.0` - GPS position tracking
- `geocoding: ^2.1.1` - Address resolution
- `permission_handler: ^11.0.1` - Permission management (existing)

## Files Modified

1. `mobile_app/lib/services/location_service.dart` (NEW)
2. `mobile_app/lib/screens/monitoring_screen.dart` (MODIFIED)
3. `mobile_app/pubspec.yaml` (MODIFIED)
4. `mobile_app/android/app/src/main/AndroidManifest.xml` (MODIFIED)
5. `mobile_app/ios/Runner/Info.plist` (MODIFIED)

## Verification

Run the following commands to verify the implementation:

```bash
# Install dependencies
cd mobile_app
flutter pub get

# Check for errors
flutter analyze

# Run on device/emulator
flutter run
```

## Notes

- Location tracking starts automatically when monitoring begins
- No additional user action required
- Works on both Android and iOS
- Gracefully degrades if location unavailable
- Does not block camera or drowsiness detection functionality
