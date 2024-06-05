import * as tf from '@tensorflow/tfjs'
import { imagenetClassesTopK, convertTfTensorToOnnxTensor, IClassInfo} from '../utils/math'
import * as ort from 'onnxruntime-web';

export async function getImageTfTensorFromPath(path: string ): Promise<tf.Tensor> {
  return new Promise((r) => {
    const src = path
    const $image = new Image();
    $image.crossOrigin = 'Anonymous';
    $image.onload = function() {
        // 将图片元素转换为Tensor
    const tensor = tf.browser.fromPixels($image)
      .resizeBilinear([224, 224]) // 更改图片大小
      .toFloat()
      .div(tf.scalar(255.0)) // 归一化
      .transpose([2, 0, 1])
      .expandDims();


      // 标准化
      const mean = tf.tensor([0.485, 0.456, 0.406]);
      const std = tf.tensor([0.229, 0.224, 0.225]);

      const normalizedTensor = tensor.sub(mean.reshape([1,3,1,1])).div(std.reshape([1,3,1,1]));
      normalizedTensor.print()
      console.log(normalizedTensor.shape)
      r(normalizedTensor);
      
    };
    $image.src = src;

  })

}

export async function inference(path: string): Promise<[IClassInfo[], number]> {
  const imageTensor = await getImageTfTensorFromPath(path);

  const preprocessedData = await convertTfTensorToOnnxTensor(imageTensor)

  const session = await ort.InferenceSession
                          .create('/mysqueezenet1_1.onnx',// /mysqueezenet1_1.onnx '/resnet.onnx',  '/squeezenet1.1-7.onnx'
                          { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
  const start = new Date();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  const outputData = await session.run(feeds);
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime())/1000;
  const output = outputData[session.outputNames[0]];
  
  const outputSoftmax = tf.softmax(tf.tensor(Array.prototype.slice.call(output.data)));
  const top5 = tf.topk(outputSoftmax, 5);
  const top5Indices = top5.indices.dataSync();
  const top5Values = top5.values.dataSync();
  const results = imagenetClassesTopK(top5Indices, top5Values);
  return [results, inferenceTime];
}
