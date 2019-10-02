# react-native-automl-vision-tflite

## Getting started

*yarn:* `$ yarn add react-native-automl-vision-tflite@git+https://git@github.com/testfirelabs/react-native-automl-vision-tflite.git`

*npm:* `$ npm install --save react-native-automl-vision-tflite@git+https://git@github.com/testfirelabs/react-native-automl-vision-tflite.git`

### Mostly automatic installation

`$ react-native link react-native-automl-vision-tflite`

### Manual installation

#### iOS

1. In XCode, in the project navigator, right click `Libraries` ➜ `Add Files to [your project's name]`
2. Go to `node_modules` ➜ `react-native-automl-vision-tflite` and add `AutomlVisionTflite.xcodeproj`
3. In XCode, in the project navigator, select your project. Add `libAutomlVisionTflite.a` to your project's `Build Phases` ➜ `Link Binary With Libraries`
4. Run your project (`Cmd+R`)<

#### Android

1. Open up `android/app/src/main/java/[...]/MainApplication.java`
  - Add `import com.reactlibrary.AutomlVisionTflitePackage;` to the imports at the top of the file
  - Add `new AutomlVisionTflitePackage()` to the list returned by the `getPackages()` method
2. Append the following lines to `android/settings.gradle`:
  	```
  	include ':react-native-automl-vision-tflite'
  	project(':react-native-automl-vision-tflite').projectDir = new File(rootProject.projectDir, 	'../node_modules/react-native-automl-vision-tflite/android')
  	```
3. Insert the following lines inside the dependencies block in `android/app/build.gradle`:
  	```
      compile project(':react-native-automl-vision-tflite')
  	```


## Usage
```javascript
import AutomlVisionTflite from 'react-native-automl-vision-tflite';

// TODO: What to do with the module?
AutomlVisionTflite;
```
