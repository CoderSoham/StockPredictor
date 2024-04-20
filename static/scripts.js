$(document).ready(function() {
    var ctx = document.getElementById('priceChart').getContext('2d');
    var chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Real-Time Price',
                borderColor: 'green',
                backgroundColor: 'green',
                fill: false,
                data: []
            }, {
                label: 'Predicted Price',
                borderColor: 'red',
                backgroundColor: 'red',
                fill: false,
                borderDash: [5, 5],
                data: []
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                }
            }
        }
    });

    $('#predictionForm').submit(function(event) {
        event.preventDefault();
        var formData = $(this).serialize();
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            dataType: 'json',
            success: function(response) {
                if (response.error) {
                    alert('Error: ' + response.error);
                } else {
                    updateChart(response.realtime_price, response.predicted_price);
                }
            },
            error: function(xhr, status, error) {
                alert('Error: ' + error);
            }
        });
    });

    function updateChart(realtimeData, predictedData) {
        var labels = Array(realtimeData.length).fill().map((_, i) => i + 1);
        chart.data.labels = labels;
        chart.data.datasets[0].data = realtimeData;
        chart.data.datasets[1].data = predictedData;
        chart.update();
    }
});
