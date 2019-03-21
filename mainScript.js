"use strict";

//Data.js

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 2/3;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.trainIndices = [];
    this.testIndices = [];
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          this.shuffledTrainIndex =
              (this.shuffledTrainIndex + 1) % this.trainIndices.length;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}

//Main Script

//Loading Data
const data = new MnistData();

async function loadData(){
  await data.load();
  alert("MNINST Dataset Loaded!");
}


const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_CHANNELS = 1;
let numberOfLayers = 0;
let opSide = IMAGE_WIDTH;
let opChannels = IMAGE_CHANNELS;
let AddingLinearLayers = false;
const NUM_OUTPUT_CLASSES = 10;

//Model for MNIST dataset
let model = tf.sequential();

async function addConvLayer(){
  let activationFunct;
  if (document.getElementById("convAct").value =='relu'){
    activationFunct = 'relu';
  } else if (document.getElementById("convAct").value =='tanh') {
    activationFunct = 'tanh';
  } else if (document.getElementById("convAct").value =='sigmoid') {
    activationFunct = 'sigmoid';
  }
  if (numberOfLayers==0) {
    model.add(tf.layers.conv2d({inputShape:[IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize : +document.getElementById("kernelSize").value,
      filters : +document.getElementById("numFilters").value,
      strides : +document.getElementById("stride").value,
      activation : activationFunct,
      kernel_initializer:tf.random_normal_initializer,
    }));
    numberOfLayers++;
  }else {
    model.add(tf.layers.conv2d({
      kernelSize : +document.getElementById("kernelSize").value,
      filters : +document.getElementById("numFilters").value,
      strides : +document.getElementById("stride").value,
      activation : activationFunct,
    }));
  }

  //Padding = 0
  opSide = Math.floor( (opSide - document.getElementById("kernelSize").value )/document.getElementById("stride").value ) + 1;
  opChannels = document.getElementById("numFilters").value;
  alert("Output size is "+opSide+"*"+opSide+"*"+opChannels);
  tfvis.show.modelSummary({name:'Model Summary'}, model);
}

async function addLinearLayer(){
  if(!AddingLinearLayers){
    model.add(tf.layers.flatten());
    AddingLinearLayers = true;
  }
  let activationFunctLinear;
  if (document.getElementById("linAct").value =='relu'){
    activationFunctLinear = 'relu';
  } else if (document.getElementById("linAct").value =='tanh') {
    activationFunctLinear = 'tanh';
  } else if (document.getElementById("linAct").value =='sigmoid') {
    activationFunctLinear = 'sigmoid';
  }
  model.add(tf.layers.dense({
    units: +document.getElementById("numUnits").value,
    activation : activationFunctLinear,
  }));
  tfvis.show.modelSummary({name:'Model Summary'}, model);
}

async function addOutputLayer() {
  if(!AddingLinearLayers){
    model.add(tf.layers.flatten());
    AddingLinearLayers = true;
  }
  model.add(tf.layers.dense({
    units : NUM_OUTPUT_CLASSES,
    activation: 'softmax',
  }));
  tfvis.show.modelSummary({name:'Model Summary'}, model);
}

async function visualize() {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  // Get the examples
  const examples = data.nextTrainBatch(1);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    //console.log(imageTensor.dataSync());
    const probarray = tf.tidy(() => {
      return examples.labels
      .slice([i, 0], [1, examples.labels.shape[1]]);
    });
    let numLabel = probarray.dataSync().indexOf(Math.max(...probarray.dataSync()));

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    surface.drawArea.appendChild(document.createTextNode("Label : " + numLabel));

    probarray.dispose();
    imageTensor.dispose();
  }
}

async function trainModel() {
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  model.fit(trainXs, trainYs, {
    batchSize: +document.getElementById("batchSize").value,
    validationData: [testXs, testYs],
    epochs: +document.getElementById("numEpochs").value,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

async function saveModel(){
  await model.save('downloads://MNIST-model')
}

//Model Loading
async function loadedModelSummary(){
  let uploadJSONInput = document.getElementById('upload-json');
  let uploadWeightsInput = document.getElementById('upload-weights');
  model = await tf.loadLayersModel(tf.io.browserFiles(
    [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
  tfvis.show.modelSummary({name:'Loaded Model Summary'}, model);
}

// Canvas script
let canvas, ctx, flag = false,
    prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    dot_flag = false, w, h;

let x = "black",
    y = 1;

function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    w = canvas.width;
    h = canvas.height;
    ctx.fillRect(0, 0, w, h);

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = x;
    ctx.lineWidth = y;
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, w, h);
    //ctx.clearRect(0, 0, w, h);
    //document.getElementById("canvasimg").style.display = "none";
}

function predictDigit() {
  let imgd = ctx.getImageData(0,0,w,h);

  // var tfImg = tf.fromPixels(imageData, 1);
  // var smalImg = tf.image.resizeBilinear(tfImg, [28, 28]);
  // smalImg = tf.cast(smalImg, 'float32');
  // var tensor = smalImg.expandDims(0);
  // tensor = tensor.div(tf.scalar(255));
  // const prediction = model.predict(tensor);
  // alert(prediction);
  // console.log(prediction.dataSync());

  let pix = imgd.data;
  let Rdata = [];
  let Gdata = [];
  let Bdata = [];
    for (var j = 0; j < 28; j++) {
      let tmp = [];
      for (var k = 0; k < 28; k++) {
        tmp.push(pix[28*j+k]);
      }
      Rdata.push(tmp);
    }
    // for (var j = 0; j < 28; j++) {
    //   let tmp = [];
    //   for (var k = 0; k < 28; k++) {
    //     tmp.push(pix[784+28*j+k]);
    //   }
    //   Gdata.push(tmp);
    // }
    // for (var j = 0; j < 28; j++) {
    //   let tmp = [];
    //   for (var k = 0; k < 28; k++) {
    //     tmp.push(pix[1568+28*j+k]);
    //   }
    //   Bdata.push(tmp);
    // }
    // let RGBimg = [Rdata,Bdata,Gdata];
    // RGBimg = tf.tensor(RGBimg);
    // RGBimg = tf.reshape(RGBimg,[28,28,3]); // 28*28*3
    // //RGBimg = tf.reshape(RGBimg,[1,3,28,28]);
    // RGBimg = RGBimg.mean(2);
    // RGBimg = tf.reshape(RGBimg,[1,28,28,1]);
    // console.log(RGBimg);
    // alert(model.predict(RGBimg).dataSync());

    let Rimg = tf.tensor(Rdata);
    Rimg = Rimg.div(tf.scalar(255));
    Rimg = tf.reshape(Rimg,[1,28,28,1]);
    console.log(Rimg.dataSync());
    let prediction = model.predict(Rimg);
    //console.log(prediction);
    let numPrediction = prediction.dataSync().indexOf(Math.max(...prediction.dataSync()));
    // let max = 0, indmax = 0;
    // for (var i = 0; i < prediction.dataSync().length; i++) {
    //   if (prediction.dataSync()[i]>max) {
    //     max = prediction.dataSync()[i];
    //     indmax=i;
    //   }
    // }
    // let numPrediction = indmax;
    alert("Predicted number : "+numPrediction + "\n" + "Probabilities = " + prediction.dataSync());
  }

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(currX, currY, 2, 2);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }
}




//Testing examples

async function visualizeTest(){
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Predicted Examples', tab: 'Testing Model'});

  // Get the examples
  const examples = data.nextTrainBatch(1);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    //Original Label
    const probarray = tf.tidy(() => {
      return examples.labels
      .slice([i, 0], [1, examples.labels.shape[1]]);
    });
    let numLabel = probarray.dataSync().indexOf(Math.max(...probarray.dataSync()));

    //Predicting
    let predictedProbArray = model.predict(tf.reshape(imageTensor,[1,28,28,1]));
    let predictedLabel = predictedProbArray.dataSync().indexOf(Math.max(...predictedProbArray.dataSync()));


    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    surface.drawArea.appendChild(document.createTextNode("Label : " + numLabel + "|| Predicted digit : " + predictedLabel));

    probarray.dispose();
    predictedProbArray.dispose();
    imageTensor.dispose();
  }
}
