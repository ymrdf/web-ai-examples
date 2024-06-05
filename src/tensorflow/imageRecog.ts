import * as tf from '@tensorflow/tfjs'
import { imagenetClassesTopK, IClassInfo} from '../utils/math'
// import { getImageTfTensorFromPath } from '../utils/imageHelper'
const IMAGE_SIZE = 224;
export async function getImageTfTensorFromPath(path: string ): Promise<tf.Tensor> {
  return new Promise((r) => {
    const src = path
    const $image = new Image();
    $image.crossOrigin = 'Anonymous';
    $image.onload = function() {

    // 将图片元素转换为Tensor
    const offset = tf.scalar(127.5);
    const tensor = tf.browser.fromPixels($image)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE]) // 更改图片大小
      .toFloat()
      .sub(offset)
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
  const model = await tf.loadGraphModel('/inceptionv3/model.json');
  const result = model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])) as tf.Tensor
  result.dispose();
  return model;
}

export async function inference(path:string):Promise<[IClassInfo[],number]> {
  const model = await loadModel();

  const imageTensor = await getImageTfTensorFromPath(path);

  const predictions = model.predict(imageTensor) as tf.Tensor

  const squeezed_tensor = tf.squeeze(predictions)

  const outputSoftmax = tf.softmax(squeezed_tensor);

  const top5 = tf.topk(outputSoftmax, 5);
  const top5Indices = top5.indices.dataSync();
  const top5Values = top5.values.dataSync();
  const results = imagenetClassesTopK(top5Indices, top5Values);
  return [results, 0.5];
}


