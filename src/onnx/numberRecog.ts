import * as tf from '@tensorflow/tfjs'
import * as ort from 'onnxruntime-web';
import { convertTfTensorToOnnxTensor, convertOnnxTensorToTfTensor } from '../utils/math'


const IMAGE_SIZE = 28;
export async function getImageTfTensorFromPath(path: string ): Promise<tf.Tensor> {
  return new Promise((r) => {
    const src = path
    const $image = new Image();
    $image.crossOrigin = 'Anonymous';
    $image.onload = function() {

    // 将图片元素转换为Tensor
    const offset = tf.scalar(255);
    const tensor = tf.browser.fromPixels($image)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE]) // 更改图片大小
      .mean(2).expandDims(-1)
      .toFloat()
      .div(offset) // 归一化
      .transpose([2, 0, 1])
      .expandDims(0);

      r(tensor);
      
    };
    $image.src = src;

  })

}


export async function inference(path:string):Promise<[Uint8Array,number]>  {
  const imageTensor = await getImageTfTensorFromPath(path);
  const preprocessedData = await convertTfTensorToOnnxTensor(imageTensor);
  const session = await ort.InferenceSession
                          .create('/numberRecog.onnx',
                          { executionProviders: ['cpu'], graphOptimizationLevel: 'all' });

  const start = new Date();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  const outputData = await session.run(feeds);
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime())/1000;
  const output = outputData[session.outputNames[0]];

  const predictions = convertOnnxTensorToTfTensor(output);

  const squeezed_tensor = tf.squeeze(predictions)
  const outputSoftmax = tf.softmax(squeezed_tensor);
  const top5 = tf.topk(outputSoftmax, 5);
  const top5Indices = top5.indices.dataSync() as Uint8Array;
  return [top5Indices, inferenceTime];
}


