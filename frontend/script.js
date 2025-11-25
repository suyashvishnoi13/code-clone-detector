let similarityChart = null;
let timeChart = null;

const API_URL = "https://code-clone-detector-1.onrender.com/analyze";

document.getElementById("checkBtn").addEventListener("click", async () => {
    const code1 = document.getElementById("code1").value.trim();
    const code2 = document.getElementById("code2").value.trim();

    if (!code1 || !code2) {
        alert("Please enter both code snippets.");
        return;
    }

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code1, code2 })
        });

        const result = await response.json();
        console.log(result);

        document.getElementById("percentage").textContent = result.overall + "%";
        document.getElementById("result").classList.remove("hidden");
        document.getElementById("detailsBtn").classList.remove("hidden");

        // store globally so charts can use
        window.cloneData = result;

    } catch (error) {
        alert("Backend error! Check API connection.");
        console.error(error);
    }
});


document.getElementById("detailsBtn").addEventListener("click", () => {
    const data = window.cloneData;
    if (!data) return;

    document.getElementById("detailsSection").classList.remove("hidden");

    // Destroy previous charts to avoid layering bugs
    if (similarityChart) similarityChart.destroy();
    if (timeChart) timeChart.destroy();

    // Similarity chart
    similarityChart = new Chart(document.getElementById("similarityChart"), {
        type: "bar",
        data: {
            labels: ["Type-1", "Type-2", "Type-3", "Type-4"],
            datasets: [{
                label: "Cloning % Similarity",
                data: [
                    data.type1,
                    data.type2,
                    data.type3,
                    data.type4
                ],
                backgroundColor: ["#00ffff", "#00c2ff", "#007bff", "#7209b7"]
            }]
        }
    });

    // Execution time chart
    timeChart = new Chart(document.getElementById("timeChart"), {
        type: "bar",
        data: {
            labels: ["Type-1", "Type-2", "Type-3", "Type-4"],
            datasets: [{
                label: "Execution Time (sec)",
                data: [
                    data.times.type1,
                    data.times.type2,
                    data.times.type3,
                    data.times.type4
                ],
                backgroundColor: ["#ffdd00", "#ffaa00", "#ff8800", "#ff5500"]
            }]
        }
    });
});
