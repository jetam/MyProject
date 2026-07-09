import { apiFetch } from "./api.js";

let predictorElement = document.getElementById("predictor");

predictorElement.addEventListener("click", async (e) => {
    e.preventDefault();

    const response = await apiFetch("/api/predictor", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ a: 5, b: 3 })
    });

    console.log( "predictor response:" );
    console.log( response );

    const res = await response.json();
    console.log( "res:" );
    console.log( res );
});
