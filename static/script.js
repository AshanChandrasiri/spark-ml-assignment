document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predictionForm").addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent page refresh

        // Get selected model
        let selectedModel = document.querySelector('input[name="model"]:checked');
        let lyrics = document.getElementById("lyrics").value.trim();

        // Validation
        if (!selectedModel) {
            alert("Please select a model.");
            return;
        }
        if (lyrics === "") {
            alert("Please enter lyrics.");
            return;
        }

        // Prepare data
        let data = {
            model: selectedModel.value,
            lyrics: lyrics
        };

        document.getElementById("predictBtn").style.display = "none";
        let progressBar = document.getElementById("progressBar");
        progressBar.style.display = "block";

        let progressFill = document.getElementById("progressFill");
        let width = 0;
        let interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
            } else {
                width += 10;
                progressFill.style.width = width + "%";
            }
        }, 300);

        // Send POST request to /predict API
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(result => {
                alert("Prediction Response: " + JSON.stringify(result))

                clearInterval(interval);
                progressFill.style.width = "100%";

                document.getElementById("predictionLabel").textContent = result.prediction_label;
                document.getElementById("predictionScore").textContent = result.prediction_score;

                // Set base64 images
                document.getElementById("barChartImg").src = result.bar_chart;
                document.getElementById("pieChartImg").src = result.pie_chart;

                // Show results section
                document.getElementById("resultSection").style.display = "block";

                // Hide progress bar after completion
                setTimeout(() => {
                    progressBar.style.display = "none";
                }, 500);
            })
            .catch(error => {
                console.error("Error:", error)

                console.error("Error:", error);
                alert("Failed to get prediction.");
                document.getElementById("predictBtn").style.display = "block";
                progressBar.style.display = "none";
            });
    });
});