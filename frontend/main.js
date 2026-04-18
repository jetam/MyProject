import * as Button from "./button.js";


console.log( "blaaa" );

let composerButton = new Button.ComposerButton( document.getElementById("composer") );

let predictorButton = new Button.PredictorButton( document.getElementById("predictor") );

let dropzone = document.getElementById("dropzone");

dropzone.addEventListener("dragover", (e) => {
    dropzone.style.background = "red";
    e.preventDefault();
});

async function upload( midiFile ) {
    const formData = new FormData();
    formData.append("midiFile", midiFile);

    console.log( "before post" );

    let response = await fetch("http://127.0.0.1:5000/api/composer/midi", {
        method: "POST",
        body: formData
    });

    const res = await response.json();
        console.log( "midiii res:" );
        console.log( res );
}

dropzone.addEventListener("drop", (e) => {
    console.log( "file dropped" );
    e.preventDefault();


    const file = e.dataTransfer.files[0];

    const formData = new FormData();
    formData.append("file", file);

    upload( file );

});

