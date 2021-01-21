let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var Obj1 = 0,
  Obj2 = 0,
  Obj3 = 0,
  Obj4 = 0,
  Obj5 = 0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output
  });
}





//function for ploting loss and accuracy starts from here...........|||||||||||
console.lineGraph = function (f, W, H, start, end, step) {
  var colors = ['red', 'orange', 'brown', 'dkgreen', 'dkcyan', 'blue', 'indigo', 'violet'];
  if (Array.isArray(f)) {
    var DATA = f;
    if (arguments.length < 4 || start == undefined) {
      start = 0;
    }
    if (arguments.length < 5 || end == undefined) {
      end = DATA.length - 1;
    }
    if (arguments.length < 6 || step == undefined) {
      step = 1;
    }
    f = function (x) {
      if (x >= DATA.length) {
        return DATA[DATA.length - 1];
      }
      if (x < 0) {
        return DATA[0];
      }
      if (x === (x | 0)) {
        return DATA[x];
      }
      var index = x | 0,
        slope = x - index;
      if (index >= DATA.length) {
        return DATA[DATA.length];
      } else
        return DATA[index] * slope + DATA[index + 1] * (1 - slope);
    }
  }

  var dw = (end - start);

  var tW = Math.ceil(2 * W / (H * 0.875));
  var style = [
    'font-size: ' + (H / 1.1666) + 'px'
  ];
  // Get the data values, and work out the min/max of the graph	
  var rawData = [];
  var min = Infinity;
  var max = -Infinity;
  //Subsample if step results in more loops than we've got width.  Also cover undefined step here.
  if (dw / step > W || !step) {
    step = dw / W;
  }

  for (var i = start; i <= end; i += step) {
    var si = Math.round((i - start) / step);
    var d = f(i);
    if (!Array.isArray(d)) {
      d = [d];
    }
    for (var p = 0; p < d.length; p += 1) {
      min = Math.min(d[p], min);
      max = Math.max(d[p], max);
    }
    rawData[si] = d;
  }

  // Generate the canvas, and get a context for it.
  var canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  var ctx = canvas.getContext('2d');

  // Set basic line styling
  ctx.fillStyle = 'none';
  ctx.lineWidth = '1px';

  //Transform and pivot the raw data, so that pointSeries[] is an array of point series
  var pointSeries = [];

  for (var i = start; i <= end; i += step) {
    var si = Math.round((i - start) / step);
    var x = Math.round((i - start) * (W - 1) / dw);
    var d = rawData[si];
    if (!Array.isArray(d)) {
      d = [d];
    }
    for (var p = 0; p < d.length; p += 1) {
      var y = (H - 1) - Math.round((d[p] - min) * (H - 1) / (max - min));
      if (!pointSeries[p]) {
        pointSeries[p] = [];
      }
      pointSeries[p].push([x, y]);
    }
  }

  //Finally, draw the graph.
  for (var ci = 0; ci < pointSeries.length; ci++) {
    var points = pointSeries[ci];
    ctx.strokeStyle = colors[ci % colors.length];
    ctx.beginPath();
    for (var pi = 0; pi < points.length; pi += 1) {
      var x = points[pi][0];
      var y = points[pi][1];
      if (pi === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
  // If the graph crosses zero, draw a line there for the X axis labels; 
  // otherwise, draw them on the base
  ctx.font = '10px monospace';
  var zero = H - 1;
  var align = 'bottom';
  if (max > 0 && min < 0) {
    zero = (H - 1) - (0 - min) * (H - 1) / (max - min);
    align = 'middle';
  }
  var sbox = ctx.measureText(start);
  var ebox = ctx.measureText(end);

  ctx.strokeStyle = 'black';
  ctx.beginPath();
  ctx.moveTo(sbox.width + 2, zero);
  ctx.lineTo(W - 3 - ebox.width, zero);
  ctx.stroke();

  ctx.textBaseline = 'middle';
  ctx.fillText(start, 0, zero);
  ctx.fillText(end, W - 1 - ebox.width, zero);


  //Draw the max and min of the graph.
  ctx.textBaseline = 'top';
  ctx.fillText(max, 0, 0);

  ctx.textBaseline = 'bottom';
  ctx.fillText(min, 0, H);

  //Add the rendered canvas to the style, and write out the pseudo-block
  style.push('background: url(' + canvas.toDataURL() + ') 0 0 no-repeat');

  console.log('%c%s',
    style.join(';'),
    new Array(tW + 1).join(' ')
  );
};
//function for ploting loss and accuracy starts from here.................||||||

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);

  model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      tf.layers.dense({
        units: 100,
        activation: "relu"
      }),
      tf.layers.dense({
        units: 5,
        activation: "softmax"
      })
    ]
  });


  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  // const optimizer = tf.train.adam(0.0001);


  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.0001),
    metrics: ["accuracy"]
  });

  model.summary();
  let loss = 0;
  let acc = 0;
  var Epoch = 0;
  var AllArrays = [];
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = parseFloat((logs.loss * 100).toFixed(4));
        acc = parseFloat((logs.acc * 100).toFixed(4));
        Epoch = Epoch + 1
        console.log("Epoch No: " + Epoch);
        console.log('LOSS: ' + loss);
        console.log('Accuracy: ' + acc);
        AllArrays.push([loss, acc]);
        if (Epoch == 10 || Epoch == 30 || Epoch == 60) {
          console.lineGraph(AllArrays, 400, 250);
        }
      }
    }
  });
}

function handleButton(elem) {
  switch (elem.id) {
    case "0":
      Obj1++;
      document.getElementById("Obj1").innerText = get1 + ":- " + Obj1;
      break;
    case "1":
      Obj2++;
      document.getElementById("Obj2").innerText = get2 + ":- " + Obj2;
      break;
    case "2":
      Obj3++;
      document.getElementById("Obj3").innerText = get3 + ":- " + Obj3;
      break;
    case "3":
      Obj4++;
      document.getElementById("Obj4").innerText = get4 + ":- " + Obj4;
      break;

    case "4":
      Obj5++;
      document.getElementById("Obj5").innerText = get5 + ":- " + Obj5;
      break;

  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "Result:- " + get1;
        break;
      case 1:
        predictionText = "Result:- " + get2;
        break;
      case 2:
        predictionText = "Result:- " + get3;
        break;
      case 3:
        predictionText = "Result:- " + get4;
        break;

      case 4:
        predictionText = "Result:- " + get5;
        break;
    }
    document.getElementById("prediction").innerText = predictionText;


    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining() {
  train();
  alert("Training Done!")
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}


init();