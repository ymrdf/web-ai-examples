import './App.css'
import ImageRecog from './components/ImageRecog'
import NumberRecog from './components/NumberRecog'
import ImageSegmention from './components/ImageSegmention'

function App() {

  return (
    <>
      <ImageRecog  width={240} height={240} />
      <NumberRecog  width={24} height={24}/>
      <ImageSegmention  width={240} height={240}/>
    </>
  )
}

export default App
