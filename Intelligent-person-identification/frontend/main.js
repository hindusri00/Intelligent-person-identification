const API_BASE = "http://127.0.0.1:5000/api";

// 1. Upload Video
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('videoInput');
    if (!fileInput.files[0]) return alert("Select a video file first!");

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.status === 'success') {
        document.getElementById('streamViewer').src = `${API_BASE}/stream?path=${encodeURIComponent(data.video_path)}`;
    }
});

// 2. Natural Language Search
document.getElementById('searchBtn').addEventListener('click', async () => {
    const query = document.getElementById('searchInput').value;
    if (!query) return;

    const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });
    const data = await res.json();

    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = `<b>Matches for "${data.query}":</b>`;

    if (data.results.length === 0) {
        resultsContainer.innerHTML += `<p>No matching individuals found.</p>`;
        return;
    }

    data.results.forEach(item => {
        resultsContainer.innerHTML += `
            <div class="result-card">
                📍 <b>@ ${item.timestamp}</b> | Person #${item.person_id}<br>
                Shirt: ${item.shirt} | Pant: ${item.pant} | Hair: ${item.hair} | Mask: ${item.mask}
            </div>`;
    });
});

// 3. Update Custom Blacklist Rules
document.getElementById('updateRulesBtn').addEventListener('click', async () => {
    const blacklist = {
        mask: document.getElementById('chkMask').checked,
        tattoo: document.getElementById('chkTattoo').checked,
        bag: document.getElementById('chkBag').checked,
        spectacles: document.getElementById('chkGlasses').checked
    };

    await fetch(`${API_BASE}/blacklist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blacklist })
    });

    alert("Alert rules updated successfully!");
});

// 4. Poll Alerts
setInterval(async () => {
    const res = await fetch(`${API_BASE}/alerts`);
    const data = await res.json();
    const alertBox = document.getElementById('alertBox');

    if (data.alerts.length > 0) {
        alertBox.innerHTML = '';
        data.alerts.forEach(alert => {
            alertBox.innerHTML += `
                <div class="alert-card">
                    ⚠️ <b>ALERT [${alert.timestamp}]</b><br>
                    ID #${alert.track_id}: ${alert.reason}
                </div>`;
        });
    } else {
        alertBox.innerHTML = '<p style="color:#94a3b8;">No threats matching active rules.</p>';
    }
}, 2500);