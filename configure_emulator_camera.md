# Configure Android Emulator to Use USB Camera

## Method 1: Using Android Studio (Easiest)

1. **Open Android Studio**
2. **Go to Tools → Device Manager**
3. **Click the ⋮ (three dots) next to your emulator**
4. **Select "Edit"**
5. **In the "Camera" section:**
   - **Front camera:** Change from "Emulated" to "Webcam0" (or "Webcam1" for external USB)
   - **Back camera:** Change from "Emulated" to "Webcam0" (or "Webcam1" for external USB)
6. **Click "Finish"**
7. **Start the emulator again**

## Method 2: Using Command Line

### Find your emulator name:
```powershell
emulator -list-avds
```

### Find available webcams:
```powershell
emulator -webcam-list
```

This will show something like:
```
List of web cameras connected to the computer:
 Camera 'webcam0' is connected to device 'Integrated Camera' on channel 0
 Camera 'webcam1' is connected to device 'USB Camera' on channel 0
```

### Start emulator with specific webcam:
```powershell
# Replace 'Medium_Phone_API_36.1' with your AVD name
# Replace 'webcam1' with your USB camera ID
emulator -avd Medium_Phone_API_36.1 -camera-front webcam1 -camera-back webcam1
```

## Method 3: Edit Config File Directly

### Location of config file:
```
C:\Users\<YourUsername>\.android\avd\<AVD_NAME>.avd\config.ini
```

### Add these lines:
```ini
hw.camera.front = webcam1
hw.camera.back = webcam1
```

## Testing

After configuration:
1. Start the emulator
2. Open the Camera app on the emulator
3. You should see your USB camera feed
4. Then run your Flutter app: `flutter run`

## Troubleshooting

If you see a black screen or error:
- Try different webcam numbers (webcam0, webcam1, webcam2)
- Make sure no other app is using the USB camera
- Restart the emulator after configuration changes
- Check that the USB camera is properly connected and recognized by Windows
