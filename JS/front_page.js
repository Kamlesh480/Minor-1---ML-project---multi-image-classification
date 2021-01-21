function othername() {
    var input1 = document.getElementById("userInput1").value;
    var input2 = document.getElementById("userInput2").value;
    var input3 = document.getElementById("userInput3").value;
    var input4 = document.getElementById("userInput4").value;
    var input5 = document.getElementById("userInput5").value;
    localStorage.setItem("storageName1", input1);
    localStorage.setItem("storageName2", input2);
    localStorage.setItem("storageName3", input3);
    localStorage.setItem("storageName4", input4);
    localStorage.setItem("storageName5", input5);
    alert(
        "Your objects have been selected, now click on start captuering picture for object"
    );
}