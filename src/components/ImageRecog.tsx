import { useRef, useState } from 'react';
import styles from './style.module.css';
import { inference } from '../tensorflow/imageRecog'
import { inference as onnxInference} from '../onnx/imageRecog'

interface Props {
  height: number;
  width: number;
}

const ImageRecog = (props: Props) => {

  const canvasRef = useRef<HTMLCanvasElement>(null);
  let image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState("");
  const [topResultConfidence, setConfidence] = useState<number>();
  const [inferenceTime, setInferenceTime] = useState("");
  
  const getImage = (pic:string) => {
    return {text:pic, value:`http://localhost:5173/${pic}.jpeg` }
  }

  const displayImageAndRunInference = (pic:string,type:string) => { 
    image = new Image();
    const sampleImage = getImage(pic);
    image.src = sampleImage.value;

    setLabel(`Inferencing...`);
    setConfidence(0);
    setInferenceTime("");

    // Draw the image on the canvas
    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    }

    if(type === "tf"){
      submitInferenceTF();
    }else{
      submitInference()
    }
  };

  const submitInference = async () => {

    const [inferenceResult,inferenceTime] = await onnxInference(image.src);

    const topResult = inferenceResult[0];

    setLabel(topResult.name.toUpperCase()+'/'+inferenceResult[1].name+"/"+inferenceResult[2].name+'/'+inferenceResult[3].name);
    setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

  };

  const submitInferenceTF = async () => {
    
    const [inferenceResult,inferenceTime] = await inference(image.src);

    const topResult = inferenceResult[0];

    setLabel(topResult.name.toUpperCase()+'/'+inferenceResult[1].name+"/"+inferenceResult[2].name+'/'+inferenceResult[3].name);
    setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);

  };

  return (
    <div>
      <h1>1,图片分类</h1>
    <button
      className={styles.grid}
      onClick={ () => displayImageAndRunInference("shark",'tf')} >
      tensorflow 识别鲨鱼
    </button>
    <button
      className={styles.grid}
      onClick={() => displayImageAndRunInference("cat",'tf')} >
      tensorflow 识别猫
    </button>
    <button
      className={styles.grid}
      onClick={ () => displayImageAndRunInference("shark",'onnx')} >
      onnx 识别鲨鱼
    </button>
    <button
      className={styles.grid}
      onClick={() => displayImageAndRunInference("cat",'onnx')} >
      onnx 识别猫
    </button>
    <br/>
    <canvas ref={canvasRef} width={props.width} height={props.height} />
    <span>{topResultLabel} {topResultConfidence}</span>
    <span>{inferenceTime}</span>
    </div>
  )
};

export default ImageRecog;