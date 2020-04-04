function show_structures(plot_path) {
  var plot_path_header = plot_path.id;
  var R_box_ = document.getElementById("R_box");
  var R1_img = document.createElement("img");
  var R1_name = "R1.png";
  var R1_path = plot_path_header.concat(R1_name);
  R1_img.src = R1_path;
  var R2_img = document.createElement("img");
  var R2_name = "R2.png";
  var R2_path = plot_path_header.concat(R2_name);
  R2_img.src = R2_path;
  R_box_.appendChild(R1_img);
  R_box_.appendChild(R2_img);
}

function showProperty(P_id) {
  var o_path = P_id.src;
  var path1 = o_path.substr(0, o_path.length - 4);
  var path2 = "y.png";
  var img_path = path1.concat(path2);
  console.log(img_path);
  document.getElementById("Y_plot").src = img_path;
  document.getElementById("P_big").src = o_path;
  var properties = P_id.id;
  var properties_list = properties.split("_");
  var dc = properties_list[0];
  var gtt = properties_list[1];
  var smile = properties_list[2];
  document.querySelector("#dc_value").textContent = dc;
  document.querySelector("#gtt_value").textContent = gtt;
  document.querySelector("#smiles").textContent = smile;
  // document.getElementById("property_plot").src =
  //   "/static/images/{{post.id}}/{{post.utctime}}P3y.png";
}
