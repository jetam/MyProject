import * as Button from "./button.js";



let composerButton = new Button.ComposerButton( document.getElementById("composer") );

let predictorButton = new Button.PredictorButton( document.getElementById("predictor") );


/* fetch("http://127.0.0.1:5000/api/add", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ a: 5, b: 3 })
})
.then(res => res.json())
.then(data => console.log(data.result)); */

/* from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
) */


    // request
// method: get, post, put, delete
// endpoint: api/sum    // this maps to fastapi route
// metadata      // tells server how to read the body
// body (optional)

    //response
// status code
// body
// headers

// fastapi-> maps http requests to python functions

// CORS prevents malicious websites calling your API
/* from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
) */