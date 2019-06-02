$(document).ready(function () {
    // Init
    console.log("New Script");
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    function showPhoto(img){
        $('#imagePreview').css('background-image', 'url(' + img + ')');
        $('#imagePreview').hide();
        $('#imagePreview').fadeIn(650);
        
    }
    
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });
    $('#selectedPhoto').change(function(){ 
        var value = $(this).val();
        var img_path = $('.thumbnail.selected img').attr("src");
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        showPhoto(img_path);
    });
    
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        console.log(form_data);
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
            error: function (request, status, error) {
                alert("Oops: Something went terribly wrong...");
                $('.loader').hide();
            }
        });
    });

});
