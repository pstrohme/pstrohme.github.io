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

    //  Slider starten
    showImage(currentIndex);

    // Der Zustand der Ampel wird aus einer JS-Datei gelesen
    // Funktion zum Aktualisieren der Ampel basierend auf dem Zustand aus der JS-Datei
    function updateTrafficLight() {
        document.querySelector(`.${window.color}`).style.opacity = 1;
    }

    // Aktualisieren der Ampel beim Laden der Seite
    updateTrafficLight();

    // Laden der JSON-Datei und Anzeigen des f1-score und des besten Modells
    fetch('../Daten/best_f1.json')
    .then(response => response.json())
    .then(data => {
        const f1ScoreElement = document.getElementById('f1-score');
        f1ScoreElement.textContent = `${data['f1-score']}`;

        const modelElement = document.getElementById('best-model');
        modelElement.textContent = `${data['model-name']}`;
    });

    // Dasselbe mit Trainingszeipunkt (hier den letzten Eintrag der Datei)
    fetch('../Daten/training_dates.json')
    .then(response => response.json())
    .then(data => {
        const trainingDates = data['training-dates'];
        const lastTrainingDate = trainingDates[trainingDates.length - 1];

        const trainingDateElement = document.getElementById('training-date');
        trainingDateElement.textContent = `${lastTrainingDate}`;
    });
    

});
