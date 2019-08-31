$(document).ready(function(){
    $('#panda_assistant').click(function(event){
        $('#chatbox').toggle();
    });

    $('#chatbox .title button').click(function(event){
        $('#chatbox').toggle();
    });
});