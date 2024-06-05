import { Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs'
import * as ort from 'onnxruntime-web';
import { convertTfTensorToOnnxTensor, convertOnnxTensorToTfTensor } from '../utils/math'

function combine(imageTensor:tf.Tensor, alphaExpanded:tf.Tensor){
   //1. 沿最后一个维度合并
  const combinedTensor = tf.concat([imageTensor, alphaExpanded], -1);

  // 第二步：将tensor转换为Uint8ClampedArray
  combinedTensor.data().then(data => {
    const clampedArray = new Uint8ClampedArray(data); 

    // 第三步：创建ImageData对象
    const imageData = new ImageData(clampedArray, 1024, 1024); 

    // 第四步：绘制到canvas
    const canvas = document.querySelector("#test") as HTMLCanvasElement;

    const ctx = canvas?.getContext('2d');
    ctx?.putImageData(imageData, 0, 0);
  });
}

export function combineImage(path:string, alphaExpanded:Tensor){

  const src = path
  const $image = new Image();
  $image.crossOrigin = 'Anonymous';
  $image.onload = function() {
      // 将图片元素转换为Tensor
    const tensor = tf.browser.fromPixels($image).resizeBilinear([1024,1024])
    const alpha4 = convertOnnxTensorToTfTensor(alphaExpanded)
    let alpha3 = tf.squeeze(alpha4, [1]);
    alpha3 = tf.reshape(alpha3, [1024, 1024, 1]);
    alpha3 = tf.mul(alpha3, 255)
    combine(tensor, alpha3)
  }
  $image.src = src

}

export async function getImageTfTensorFromPath(path: string ): Promise<tf.Tensor> {
  return new Promise((r) => {
    const src = path
    const $image = new Image();
    $image.crossOrigin = 'Anonymous';
    $image.onload = function() {
        // 将图片元素转换为Tensor
    const tensor = tf.browser.fromPixels($image)
      .resizeBilinear([1024,1024]) // 更改图片大小
      .toFloat()
      .div(tf.scalar(255.0)) // 归一化
      .transpose([2, 0, 1])
      .expandDims();
      // 标准化
      const mean = tf.tensor([0.5,0.5,0.5]);
      const std = tf.tensor([1.0,1.0,1.0]);
      const normalizedTensor = tensor.sub(mean.reshape([1,3,1,1])).div(std.reshape([1,3,1,1]));
      normalizedTensor.print()
      console.log(normalizedTensor.shape)
      r(normalizedTensor);
      
    };
    $image.src = src;
  })
}

async function runInference(session: ort.InferenceSession, preprocessedData: Tensor): Promise<Tensor> {
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  const outputData = await session.run(feeds);
  const output = outputData[session.outputNames[0]];

  return output
}

export async function runModel(preprocessedData: Tensor): Promise<Tensor> {

  const session = await ort.InferenceSession
                          .create('/rmbg.onnx', 
                          { executionProviders: ['cpu'], graphOptimizationLevel: 'all' });
  const out =  await runInference(session, preprocessedData);
  return  out;
}

export async function inference(path: string): Promise<Tensor> {
  const res = await getImageTfTensorFromPath(path)
  const imageTensor = await convertTfTensorToOnnxTensor(res)
  const out = await runModel(imageTensor);
  combineImage(path, out)
  return out;
}