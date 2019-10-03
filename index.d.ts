export interface Prediction {
  confidence: number;
  label: string;
}

declare module 'react-native-automl-vision-tflite' {
  function loadModel(
    modelFilePath: string,
    labelsFilePath: string,
    callback: (error: string, modelId: string) => void
  ): void;

  function runModelOnImage(
    modelId: string,
    imageFilePath: string,
    numberOfResults: number,
    threshold: number,
    callback: (error: string, result: Prediction[]) => void
  ): void;

  function close(modelId: string): void;
}
