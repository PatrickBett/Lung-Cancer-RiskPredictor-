document.addEventListener("DOMContentLoaded", function () {
  const predictionForm = document.getElementById("prediction-form");
  const resultContainer = document.getElementById("result-container");
  const loading = document.getElementById("loading");
  const resultHeading = document.getElementById("result-heading");
  const predictionText = document.getElementById("prediction-text");
  const probabilityText = document.getElementById("probability-text");
  const riskLevelText = document.getElementById("risk-level");
  const resultItem = document.getElementById("result-item");

  predictionForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Show loading spinner
    loading.style.display = "block";
    resultContainer.style.display = "none";

    try {
      const formData = new FormData(predictionForm);
      const response = await fetch("/predict/", {
        method: "POST",
        body: formData,
        headers: {
          "X-Requested-With": "XMLHttpRequest",
        },
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      // Update the result container with prediction results
      if (data.prediction === 1) {
        resultHeading.textContent =
          "Risk Assessment: Potential Lung Cancer Risk";
        predictionText.textContent =
          "The model predicts a potential risk of lung cancer.";
      } else {
        resultHeading.textContent = "Risk Assessment: Low Lung Cancer Risk";
        predictionText.textContent =
          "The model predicts a low risk of lung cancer.";
      }

      // Update probability and risk level
      probabilityText.textContent = `Probability: ${(
        data.probability * 100
      ).toFixed(2)}%`;
      riskLevelText.textContent = `Risk Level: ${data.risk_level}`;

      // Add appropriate CSS class for risk level
      resultItem.className = "result-item";
      resultItem.classList.add(`risk-${data.risk_level.toLowerCase()}`);

      // Show the result container
      resultContainer.style.display = "block";
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while making the prediction. Please try again.");
    } finally {
      // Hide loading spinner
      loading.style.display = "none";
    }
  });
});
