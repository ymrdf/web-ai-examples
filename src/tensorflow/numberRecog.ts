import * as tf from '@tensorflow/tfjs'
// import { getImageTfTensorFromPath } from '../utils/imageHelper'
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
      // .transpose([2, 0, 1])
      .expandDims(0);

      r(tensor);
      
    };
    $image.src = src;

  })

}


async function loadModel() {
  // 使用tf.loadLayersModel 或loadGraphModel加载转换后的模型
  const model = await tf.loadGraphModel('/numberRecogV1/model.json');
  const result = model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])) as tf.Tensor
  result.dispose();
  return model;
}

export async function inference(path:string):Promise<[Uint8Array,number]> {
  const model = await loadModel();

  const imageTensor = await getImageTfTensorFromPath(path);

  const predictions = model.predict(imageTensor) as tf.Tensor

  const squeezed_tensor = tf.squeeze(predictions)
  const outputSoftmax = tf.softmax(squeezed_tensor);
  const top5 = tf.topk(outputSoftmax, 5);
  const top5Indices = top5.indices.dataSync() as Uint8Array;
  return [top5Indices, 0.5];
}


