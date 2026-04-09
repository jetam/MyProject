
class Button
{
    constructor( button ){
        this.button = button;
        this.sLog = String();

        console.log( "in construcot" );

        this.bValid = false;
        if( button ) {

            button.addEventListener( "click", ( e ) => { // if function(e) this refers to the dom element!
                console.log( "bla1" );
                e.preventDefault();

                console.log( "bla" );
                this.logClick();

                this.httpRequest();
                
            } );

            this.bValid = true;
        }
    };

    logClick() {
        console.log( this.sLog );
    }

}

export class PredictorButton extends Button
{
    constructor( button ) {
        super( button );
        this.sLog = logValues.PREDICTOR;
    }

    async httpRequest() {       

        const response = await fetch("http://127.0.0.1:5000/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ a: 5, b: 3 })
            })

        console.log( "predictor respose:" );
        console.log( response );

        const res = await response.json();
        console.log( "res:" );
        console.log( res );
    }

}

export class ComposerButton extends Button
{
    constructor( button ) {
        super( button );
        this.sLog = logValues.COMPOSER;
    }

    async httpRequest() {       

        const response = await fetch("http://127.0.0.1:5000/api/compose", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ a: 5, b: 3 })
            })

        console.log( "composer respose:" );
        console.log( response );

        const res = await response.json();
        console.log( "res:" );
        console.log( res );
    }

}

export const logValues = {
    PREDICTOR: "predictor button pressed!",
    COMPOSER: "composer button pressed!"
}