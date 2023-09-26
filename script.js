document.addEventListener("DOMContentLoaded", function () {

    // Slider Funktionen

    const sliderImages = document.querySelectorAll(".slider img");
    let currentIndex = 0;

    function showImage(index) {
        sliderImages.forEach((img, i) => {
            if (i === index) {
                img.style.display = "block";
            } else {
                img.style.display = "none";
            }
        });
    }

    function nextImage() {
        currentIndex = (currentIndex + 1) % sliderImages.length;
        showImage(currentIndex);
    }

    function prevImage() {
        currentIndex = (currentIndex - 1 + sliderImages.length) % sliderImages.length;
        showImage(currentIndex);
    }

    var interval = setInterval(nextImage, 10000); // Wechsel alle 10 Sekunden

    // Manuell zwischen Bildern wechseln
    const nextButton = document.getElementById("nextButton");
    const prevButton = document.getElementById("prevButton");

    nextButton.addEventListener("click", ()=> {
        nextImage();
        clearInterval(interval)
        interval = setInterval(nextImage, 10000);
    });
    prevButton.addEventListener("click", ()=> {
        prevImage();
        clearInterval(interval)
        interval = setInterval(nextImage, 10000);
    });

    // Starten Sie den Slider
    showImage(currentIndex);

    // Annahme: Der Zustand der Ampel wird aus einer JSON-Datei gelesen
    // Beispielinhalt der JSON-Datei: {"color": "red"}


    // Funktion zum Aktualisieren der Ampel basierend auf dem Zustand aus der JSON-Datei
    function updateTrafficLight() {
        document.querySelector(`.${window.color}`).style.opacity = 1;
    }

    // Aktualisieren der Ampel beim Laden der Seite
    updateTrafficLight();

});
