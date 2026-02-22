console.log("utils.js loaded!");

const clearBtn = document.getElementById('clearBtn');
const subBtn = document.getElementById('submit');
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

const results = document.getElementById('results');

canvas.addEventListener('mousedown', () => { drawing = true; ctx.beginPath(); });
canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('touchmove', drawTouch, {passive: false});

clearBtn.addEventListener('click', resetCanvas);
subBtn.addEventListener('click', sendImage);

window.onload = function () {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
};

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 10;
    ctx.lineCap = 'square';
    ctx.strokeStyle = 'white';

    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function drawTouch(e) {
    e.preventDefault();
    ctx.lineWidth = 10;
    ctx.lineCap = 'square';
    ctx.strokeStyle = 'white';

    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
        

async function sendImage() {
    const modelResultDiv = document.createElement("div");
    
    const dataURL = canvas.toDataURL();
    const res = await fetch('/classify', {
        method: 'POST',
        body: JSON.stringify({image: dataURL}),
        headers: {'Content-Type': 'application/json'}
    });
            
    const modelRes = await res.json();
    
    const container = document.createElement("div");
    container.className = 'container';

    const graphElem = document.createElement("div");
    graphElem.className = 'box';
    // graphElem.style.width='400px';
    // graphElem.style.height='150px';
            

    const img = document.createElement('img');
    img.src = dataURL; // Set it as the image src
    img.className = 'box';
            

    const igImg = document.createElement('div');
    igImg.className = 'box';
    // igImg.style.width='300px';
    // igImg.style.height='300px';
            

    var data = {
        x: modelRes.labels,
        y: modelRes.values,
        type: 'bar',
    };

    var layout = {
        autosize: true,
        width: 400,
        height: 300,
        xaxis: {
            title: {text: 'Class'},
            tickmode: 'array',
            tickvals: modelRes.labels,
            ticktext: modelRes.labels
        },
        yaxis: {
            title: {text: 'Softmax Prob.'},
            range: [0,1],
        },
    };

    Plotly.newPlot(graphElem, [data], layout);
            
    var colourscaleValues = [
        [-1, '#3D9970'],
        [1, '#3D9970']
    ];

    var igData = {
        z: modelRes.igimage,
        type: 'heatmap',
        zmin:-1,
        zmax: 1
        
    };

    const igLayout = {
        autosize: true,
        width: 400,
        height: 400,
        xaxis: {
            showticklabels: false,
            ticks: '',
            scaleanchor: 'y',
            constrain: 'domain', 
        },
        yaxis: {
            showticklabels: false,
            ticks: ''
        }
    };

    Plotly.newPlot(igImg, [igData], igLayout);

    container.appendChild(img);
    container.appendChild(graphElem);
    container.appendChild(igImg);

    results.prepend(container);

    resetCanvas();
            
}