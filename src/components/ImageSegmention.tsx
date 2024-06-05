import { useRef, useState } from 'react';
import { inference } from '../rmbg/predict';
import styles from './style.module.css';

interface Props {
  height: number;
  width: number;
}


const ImageRecog = (props: Props) => {

  const canvasRef = useRef<HTMLCanvasElement>(null);
  let image: HTMLImageElement;
  const [topResultLabel, setLabel] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");
  
  const getImage = () => {
    return {text:"cat", value:`http://localhost:5173/giraffe.jpg` }
  }

  const displayImageAndRunInference = () => { 
    image = new Image();
    const sampleImage = getImage();
    image.src = sampleImage.value;

    setLabel(`Inferencing...`);
    setInferenceTime("");

    const canvas = canvasRef.current;
    const ctx = canvas!.getContext('2d');
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    }
    submitInference();
  };

  const submitInference = async () => {
    await inference(image.src);
  };

  return (
    <div>
      <h1>3,物体分割</h1>
    <div>
    <button
      className={styles.grid}
      onClick={ () => displayImageAndRunInference()} >
      分割
    </button>
    <br/>
    <canvas ref={canvasRef} width={props.width} height={props.width} />
    <span>{topResultLabel}</span>
    <span>{inferenceTime}</span>
    </div>
    <div>
      <canvas id="test" width={1024} height={1024} />
    </div>
    </div>
  )
};

export default ImageRecog;