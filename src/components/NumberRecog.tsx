import { useRef, useState } from 'react';
import styles from './style.module.css';
import { inference } from '../tensorflow/numberRecog'
import { inference as onnxInference } from '../onnx/numberRecog'

interface Props {
  height: number;
  width: number;
}

const ImageRecog = (props: Props) => {

  const canvasRef = useRef<HTMLCanvasElement>(null);
  let image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState<number|null>(null);
  const [inferenceTime, setInferenceTime] = useState("");
  
  const getImage = () => {
    const number = Math.floor(Math.random() * 10)
    return {text:number, value:`http://localhost:5173/number_images/m${number}.png` }
  }

  const displayImageAndRunInference = (type:string) => { 
    image = new Image();
    const sampleImage = getImage();
    image.src = sampleImage.value;

    setLabel(null);
    setInferenceTime("");

    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    }
    if(type === 'tf'){
      submitInferenceTF();
    }else{
      submitInference()
    }
  };


  const submitInferenceTF = async () => {
    const [inferenceResult,inferenceTime] = await inference(image.src);

    setLabel(inferenceResult[0]);
    setInferenceTime(`   时间: ${inferenceTime} seconds`);

  };

  const submitInference = async () => {
    const [inferenceResult,inferenceTime] = await onnxInference(image.src);

    setLabel(inferenceResult[0]);
    setInferenceTime(`   时间: ${inferenceTime} seconds`);

  };

  return (
    <div>
      <h1>2,自训练模型</h1>
    <button
      className={styles.grid}
      onClick={ () => displayImageAndRunInference('tf')} >
      tensorflow识别数字
    </button>
    <button
      className={styles.grid}
      onClick={ () => displayImageAndRunInference('onnx')} >
      onnx 识别数字
    </button>
    <br/>
    <br />
    <canvas ref={canvasRef} width={props.width} height={props.height} />
    <span>{topResultLabel}</span>
    <span>{inferenceTime}</span>
    </div>
  )
};

export default ImageRecog;