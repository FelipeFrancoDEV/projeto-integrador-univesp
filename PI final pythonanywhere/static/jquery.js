jQuery(document).ready(function($) {

  $('#show_password').click(function(e) {
    e.preventDefault();
    if ( $('#senha').attr('type') == 'password' ) {
      $('#senha').attr('type', 'text');
      $('#show_password').attr('class', 'fa fa-eye');
    } else {
        $('#senha').attr('type', 'password');
        $('#show_password').attr('class', 'fa fa-eye-slash');
    }
  });

  $('#show_password_cad').click(function(e) {
    e.preventDefault();
    if ( $('#senha_cad').attr('type') == 'password' ) {
      $('#senha_cad').attr('type', 'text');
      $('#show_password_cad').attr('class', 'fa fa-eye');
    } else {
        $('#senha_cad').attr('type', 'password');
        $('#show_password_cad').attr('class', 'fa fa-eye-slash');
    }
  });

});