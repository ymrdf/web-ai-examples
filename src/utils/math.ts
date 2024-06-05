import {labels} from './constants'
import { Tensor,  } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs'


export interface IClassInfo{
  id: string,
  index: number,
  name: string,
  probability: number
}

/**
 * 找到最大可能性的几个值
 */
export function imagenetClassesTopK(topIndices:Float32Array| Uint8Array | Int32Array, classProbabilities:Float32Array| Uint8Array | Int32Array):IClassInfo[] {
  const result = []
  for(let i = 0; i<topIndices.length; i++){
    const probIndex = topIndices[i]
    const iClass = labels[probIndex];
    result.push({
      id: iClass,
      index: probIndex,
      name: iClass.replace(/_/g, ' '),
      probability: classProbabilities[i]
    }) ;
  }
  return result
}

export async function convertTfTensorToOnnxTensor(tfTensor:tf.Tensor) {
  // 获取tfTensor的数据。data()方法返回一个承诺，它将解析为包含tfTensor所有值的TypedArray。
  const data = await tfTensor.data();
  
  // 获取tfTensor形状
  const shape = tfTensor.shape;
  
  // 获取数据类型，然后转换为onnxruntime-web所需的格式
  // tfTensor.dtype返回如"float32"或"int32"字符串，需根据实际情况调整
  let dataType:keyof Tensor.DataTypeMap = 'float32'; // 这里默认为'float32'，根据你的Tensor实际数据类型进行调整
  switch (tfTensor.dtype) {
      case 'float32':
          dataType = 'float32';
          break;
      case 'int32':
          // onnxruntime-web在某些版本中可能不支持int32，需按实际情况调整
          dataType = 'int32';
          break;
      // 这里可以添加更多数据类型的处理
      default:
          throw new Error(`Unsupported data type: ${tfTensor.dtype}`);
  }
  
  // 创建onnxruntime-web的Tensor
  const onnxTensor = new Tensor(dataType, data,  shape);
  return onnxTensor;
}

export function convertOnnxTensorToTfTensor(onnxTensor:Tensor) {
  // 首先获取onnxTensor的数据
  const data = onnxTensor.data;
  // 通过观察或已知获取的onnxTensor的形状
  const shape = onnxTensor.dims;
  // 使用@tensorflow/tfjs的tensor方法，根据数据和形状创建新的tensor
  const tfTensor = tf.tensor(data as Float32Array, shape as [number, number,number,number]);
  return tfTensor;
}
