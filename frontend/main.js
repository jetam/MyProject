import * as Button from "./button.js";

let composerElement = document.getElementById("generate");

let composerButton = new Button.ComposerButton( composerElement );

let predictorButton = new Button.PredictorButton( document.getElementById("predictor") );

let dropzone = document.getElementById("dropzone");

let modelSelect = document.getElementById("modelSelect");

let downloadButton = document.getElementById("downloadMidi");

// ordered top-to-bottom: hiding a step should hide everything after it too
const steps = [modelSelect, composerElement, downloadButton];

function hideFrom(step) {
    const idx = steps.indexOf(step);
    for (let i = idx; i < steps.length; i++) {
        steps[i].classList.add("d-none");
    }
}

let currentFileName = "";

composerButton.onGenerated = (blob) => {
    downloadButton.classList.remove("d-none");
    downloadButton.onclick = () => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "generated.mid";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
};

dropzone.addEventListener("dragover", (e) => {
    dropzone.style.background = "purple";
    e.preventDefault();
});

async function uploadMIDI( midiFile ) {
    const formData = new FormData();
    formData.append("midiFile", midiFile);

    console.log( "before post" );

    let response = await fetch("http://127.0.0.1:5000/api/music/midi/upload", {
        method: "POST",
        body: formData // todo: also send selected model
    });

    const res = await response.json();
        console.log( "midiii res:" );
        console.log( res );
}

async function fetchModels() {
    let response = await fetch("http://127.0.0.1:5000/api/music/model/selection");
    const res = await response.json();

    modelSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select model";
    placeholder.selected = true;
    modelSelect.appendChild(placeholder);

    for (const model of res.models) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    }
}

composerElement.addEventListener("click", () => {
    hideFrom(downloadButton);
});

modelSelect.addEventListener("change", async (e) => {
    hideFrom(composerElement);

    dropzone.innerHTML = `
        <div class="spinner-border text-light" role="status"></div>
        <span class="ms-2">fine tuning...</span>
    `;

    await fetch("http://127.0.0.1:5000/api/music/model/select", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: e.target.value })
    });

    const placeholder = modelSelect.querySelector('option[value=""]');
    if (placeholder) placeholder.remove();

    composerElement.classList.remove("d-none");
    dropzone.textContent = currentFileName;
});

dropzone.addEventListener("drop", async (e) => {
    console.log( "file dropped" );
    e.preventDefault();


    const file = e.dataTransfer.files[0];

    const formData = new FormData();
    formData.append("file", file);

    hideFrom(modelSelect);

    dropzone.innerHTML = `
        <div class="spinner-border text-light" role="status"></div>
        <span class="ms-2">fine tuning...</span>
    `;

    await uploadMIDI( file );
    await fetchModels();
    modelSelect.classList.remove("d-none");
    currentFileName = file.name;
    dropzone.textContent = currentFileName;
});

