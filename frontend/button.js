
class Button
{
    constructor( button ){
        this.button = button;
        this.sLog = String();

        console.log( "in construcot" );

        this.bValid = false;
        if( button ) {

            button.addEventListener( "click", (e) => { // if function(e) this refers to the dom element!
                console.log( "bla1" );
                e.preventDefault();

                console.log( "bla" );
                this.logClick();
                
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

}

export class ComposerButton extends Button
{
    constructor( button ) {
        super( button );
        this.sLog = logValues.COMPOSER;
    }

}

export const logValues = {
    PREDICTOR: "predictor button pressed!",
    COMPOSER: "composer button pressed!"
}