function toggleSidebar() {
    let sidebar = document.getElementById("sidebar");
    let main = document.querySelector(".main");

    sidebar.classList.toggle("collapsed");
    main.classList.toggle("collapsed");
}

// alert
function triggerAlert(id){
    fetch("/fake_alert/" + id)
    .then(() => location.reload());
}

// chart
function initChart(labels, values){
    let ctx = document.getElementById("chart");

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Alerts",
                data: values
            }]
        }
    });
}