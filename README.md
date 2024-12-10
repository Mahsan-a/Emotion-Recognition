# flutter_gesture_tflite

A Flutter example application for running a tflite model


## Getting Started

!Attention!
1) need to root an emulator

-after running an emulator, go to Terminal on android studio and go to a folder where you have adb.exe using cd command

Windows 10
https://stackoverflow.com/questions/35854238/where-is-adb-exe-in-windows-10-located

cd C:\Users\[user]\AppData\Local\Android\sdk\platform-tools

Mac
https://stackoverflow.com/questions/36298696/run-adb-shell-on-os-x/36298750

cd /Users/[user-name]/Library/Android/sdk/platform-tools

then type following commands

$ > adb shell

generic_x86:/ $

generic_x86:/ $ exit

$ > adb root

restarting adbd as root

$ > adb shell

generic_x86:/ #

2)
 - Make sure you have Android studio version that supports TFlite models
 - We suggest to explore this library: https://pub.dev/packages/tflite_flutter which has been
    used in this template for model inference. But feel free to use any other package to suport your tflite models
   in flutter
   - Check the installation instruction for the library. You need to run a script at the root of your flutter project to enable tflite support
   