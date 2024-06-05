# web-ai-examples
The repo is a example to introduces how to run machine learning models in front-end development, focusing on two main methods: TensorFlow.js and OnnxRuntime-web. First, we need to save or convert models into the appropriate format, and then use the corresponding tools to load and execute them.    

TensorFlow.js allows running TensorFlow framework models directly in the browser, providing developers with a convenient way to train and deploy models. However, the existing resources for TensorFlow.js are relatively limited, and its conversion tool, tensorflowjs_wizard, can be challenging to use.    

In contrast, OnnxRuntime-web supports loading ONNX models trained with various frameworks, offering greater flexibility. However, due to the limitations of WebGL, some operators may not be supported, requiring reliance on Assembly to ensure model compatibility, which may impact execution speed.     

For data processing, the tensor calculation methods provided by OnnxRuntime-web are relatively few. We can use methods from @tensorflow/tfjs for data processing and then convert the processed results into tensor objects supported by OnnxRuntime-web for further application.


# run

```
npm run dev
```
