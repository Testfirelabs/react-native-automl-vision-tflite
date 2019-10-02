'use strict';

/*
    Example React Native App (tested with 0.61)
    Create a new app with `react-native init ExampleApp`
    Add react-native-camera (including the permissions and build.gradle changes)
    Replace index.js with this file
    Add the following to android/app/build.gradle in the android section above defaultConfig

        aaptOptions {
            noCompress 'tflite'
        }
    
    Copy your tflite model and labels files to android/app/src/main/assets and set the path in app.json
        NOTE: path should be relative to the assets folder
 */

import React, { PureComponent } from 'react';
import { AppRegistry, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { RNCamera } from 'react-native-camera';

import { name as appName, modelFile, labelsFile } from './app.json';

import AutomlVisionTflite from 'react-native-automl-vision-tflite';

const PendingView = () => (
    <View
        style={{
            flex: 1,
            backgroundColor: 'lightgreen',
            justifyContent: 'center',
            alignItems: 'center',
        }}
    >
        <Text>Waiting</Text>
    </View>
);

class ExampleApp extends PureComponent {
    componentDidMount() {
        // this can still be improved 
        // SEE https://react-native-community.github.io/react-native-camera/docs/react-navigation
        AutomlVisionTflite.loadModel(modelFile, labelsFile, (err, res) => {
            if (err)
                console.log(`Failed to load AutoML model: ${err}`);
            else
                this.setState({ autoMlModelId: res });
        });
    }

    componentWillUnmount() {
        if (this.state.autoMlModelId) {
            AutomlVisionTflite.close(this.state.autoMlModelId);
        }
    }

    render() {
        // nearly verbatim from the react-native-camera example
        // SEE https://react-native-community.github.io/react-native-camera/docs/rncamera
        return (
            <View style={styles.container}>
                <RNCamera
                    ref={ref => {
                        this.camera = ref;
                    }}
                    style={styles.preview}
                    type={RNCamera.Constants.Type.back}
                    flashMode={RNCamera.Constants.FlashMode.auto}
                    androidCameraPermissionOptions={{
                        title: 'Permission to use camera',
                        message: 'We need your permission to use your camera',
                        buttonPositive: 'Ok',
                        buttonNegative: 'Cancel',
                    }}
                    captureAudio={false}
                />
                <View style={{ flex: 0, flexDirection: 'row', justifyContent: 'center' }}>
                    <TouchableOpacity onPress={this.takePicture.bind(this)} style={styles.capture}>
                        <Text style={{ fontSize: 14 }}> SNAP </Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    takePicture = async () => {
        if (this.camera) {
            const options = { quality: 0.5 };
            const data = await this.camera.takePictureAsync(options);
            console.log(data.uri);
            AutomlVisionTflite.runModelOnImage(autoMlModelId, data.uri, 3, 0.1, (err, res) => {
                if (err)
                    console.log(`Failed to run AutoML model: ${err}`);
                else
                    console.log(res);
            });
        }
    };
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        flexDirection: 'column',
        backgroundColor: 'black',
    },
    preview: {
        flex: 1,
        justifyContent: 'flex-end',
        alignItems: 'center',
    },
    capture: {
        flex: 0,
        backgroundColor: '#fff',
        borderRadius: 5,
        padding: 15,
        paddingHorizontal: 20,
        alignSelf: 'center',
        margin: 20,
    },
});

AppRegistry.registerComponent(appName, () => ExampleApp);