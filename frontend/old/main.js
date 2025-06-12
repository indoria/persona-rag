window.addEventListener('DOMContentLoaded', async () => {
    const journalistSelect = document.getElementById('journalist');
    const outputDiv = document.getElementById('output');
    // Fetch journalist list
    const resp = await fetch('/journalists');
    const journalists = await resp.json();
    for (const j of journalists) {
        const opt = document.createElement('option');
        opt.value = j.id;
        opt.textContent = `${j.name} — ${j.bio}`;
        journalistSelect.appendChild(opt);
    }
    document.getElementById('submit').onclick = async () => {
        outputDiv.textContent = "Generating response...";
        const journalist_id = journalistSelect.value;
        const pitch_text = document.getElementById('pitch').value;
        const resp = await fetch('/generate_response', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({journalist_id, pitch_text}),
        });
        const data = await resp.json();
        if (data.error) {
            outputDiv.innerHTML = `<span class="text-red-600">${data.error}</span>`;
        } else {
            outputDiv.innerHTML = `<strong>Persona Response:</strong><br>${data.response.replace(/\n/g, '<br>')}`;
        }
    };
});