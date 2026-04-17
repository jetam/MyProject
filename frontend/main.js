import * as Button from "./button.js";


console.log( "blaaa" );

let composerButton = new Button.ComposerButton( document.getElementById("composer") );

let predictorButton = new Button.PredictorButton( document.getElementById("predictor") );

let dropzone = document.getElementById("dropzone");

dropzone.addEventListener("dragover", (e) => {
    dropzone.style.background = "red";
    e.preventDefault();
});

dropzone.addEventListener("drop", (e) => {
    console.log( "file dropped" );
    e.preventDefault();

    const files = e.dataTransfer.files;
    const file = files[0];

    console.log("Dropped file:", file);

    const reader = new FileReader();

    reader.onload = (event) => {
        const data = event.target.result;
        console.log("File content in RAM:", data);
        dropzone.style.background = "#0b5ed7";
    };

    reader.readAsArrayBuffer(file);
});

