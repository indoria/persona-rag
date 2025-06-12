document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const htmlElement = document.documentElement; // For theme class
    const personaList = document.getElementById('persona-list');
    const pitchTextarea = document.getElementById('pitch-text');
    const sendPitchBtn = document.getElementById('send-pitch-btn');
    const responsesContainer = document.getElementById('responses-container');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const pitchStatusMessage = document.getElementById('pitch-status-message');

    // --- Application State ---
    let selectedPersonaIds = new Set();
    let currentTheme = 'light';
    let mockJournalists = [];

    const fetchJournalists = async () => {
        try {
            const response = await fetch('/journalists'); // Make the GET request to /journalists
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error("Error fetching journalists:", error);
            return [];
        }
    };

    const handlePersonaClick = (personaElement, personaId) => {
        if (selectedPersonaIds.has(personaId)) {
            selectedPersonaIds.delete(personaId);
            personaElement.classList.remove('selected');
            personaElement.setAttribute('aria-checked', 'false');
        } else {
            selectedPersonaIds.add(personaId);
            personaElement.classList.add('selected');
            personaElement.setAttribute('aria-checked', 'true');
        }
        // console.log('Selected Personas:', Array.from(selectedPersonaIds));
    };

    const renderPersonas = (journalists) => {
        personaList.innerHTML = ''; // Clear existing content
        journalists.forEach(persona => {
            const personaItem = document.createElement('div');
            personaItem.className = 'persona-item';
            personaItem.setAttribute('data-id', persona.id);
            personaItem.setAttribute('tabindex', '0'); // Make div focusable for keyboard navigation
            personaItem.setAttribute('role', 'checkbox'); // For accessibility
            personaItem.setAttribute('aria-checked', selectedPersonaIds.has(persona.id) ? 'true' : 'false');

            if (selectedPersonaIds.has(persona.id)) {
                personaItem.classList.add('selected');
            }

            personaItem.innerHTML = `
                <img src="${persona.pic}" alt="${persona.name} avatar" class="persona-avatar">
                <span class="persona-name">${persona.name}</span>
                <span class="persona-role">${persona.role}</span>
                <i class="fas fa-check-circle selected-icon"></i>
            `;

            personaItem.addEventListener('click', () => handlePersonaClick(personaItem, persona.id));
            personaItem.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault(); // Prevent default scroll for spacebar
                    handlePersonaClick(personaItem, persona.id);
                }
            });

            personaList.appendChild(personaItem);
        });
    };

    const generateResponse = (pitch_text, journalist_ids) => {
        if (!pitch_text || journalist_ids.length === 0) {
            reject("Please provide pitch text and select at least one persona.");
            return;
        }
        return new Promise((resolve, reject) => {
            setTimeout(() => {

                const responses = journalist_ids.map(id => {
                    const persona = mockJournalists.find(j => j.id === id);
                    if (persona) {
                        return `**Response from ${persona.name} (${persona.role}):**\n"Your pitch about '${pitch_text.substring(0, 50)}...' resonates with my expertise. I can offer insights from the perspective of a ${persona.role} on how to develop this further, focusing on scalability and ethical considerations."`;
                    }
                    return `Error: Persona with ID ${id} not found.`;
                });
                resolve(responses);
            }, 1500); // Simulate longer processing time for response generation
        });
    };
    
    const renderResponses = (responses) => {
        responsesContainer.innerHTML = '';
        if (responses.length === 0) {
            responsesContainer.innerHTML = '<p class="status-message no-responses">No responses generated or available.</p>';
            return;
        }
        responses.forEach(response => {
            const responseCard = document.createElement('div');
            responseCard.className = 'response-card';
            // Simple Markdown-like bolding for demo
            const formattedResponse = response.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            responseCard.innerHTML = `<p>${formattedResponse.replace(/\n/g, '<br>')}</p>`;
            responsesContainer.appendChild(responseCard);
        });
    };

    const handleSendPitch = async () => {
        const pitchText = pitchTextarea.value.trim();
        const personaIdsArray = Array.from(selectedPersonaIds);

        // Basic validation
        if (pitchText.length === 0) {
            pitchStatusMessage.textContent = 'Please enter some text for your pitch.';
            pitchStatusMessage.classList.remove('loading', 'success');
            pitchStatusMessage.classList.add('error');
            return;
        }
        if (personaIdsArray.length === 0) {
            pitchStatusMessage.textContent = 'Please select at least one AI persona.';
            pitchStatusMessage.classList.remove('loading', 'success');
            pitchStatusMessage.classList.add('error');
            return;
        }

        // Set loading state
        sendPitchBtn.disabled = true;
        sendPitchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
        pitchStatusMessage.textContent = 'Generating responses...';
        pitchStatusMessage.classList.remove('error', 'success');
        pitchStatusMessage.classList.add('loading');
        responsesContainer.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Fetching responses...</div>';

        try {
            const responses = await generateResponse(pitchText, personaIdsArray);
            renderResponses(responses);
            pitchStatusMessage.textContent = 'Responses generated successfully!';
            pitchStatusMessage.classList.remove('error', 'loading');
            pitchStatusMessage.classList.add('success');
        } catch (error) {
            console.error('Error generating responses:', error);
            responsesContainer.innerHTML = `<p class="status-message error">Error: ${error}. Please try again.</p>`;
            pitchStatusMessage.textContent = 'Failed to generate responses.';
            pitchStatusMessage.classList.remove('loading', 'success');
            pitchStatusMessage.classList.add('error');
        } finally {
            // Reset button and status
            sendPitchBtn.disabled = false;
            sendPitchBtn.innerHTML = 'Send Pitch <i class="fas fa-paper-plane"></i>';
            // Clear status message after a short delay if successful
            if (pitchStatusMessage.classList.contains('success')) {
                setTimeout(() => {
                    pitchStatusMessage.textContent = '';
                    pitchStatusMessage.classList.remove('success');
                }, 3000);
            }
        }
    };

    const applyTheme = (theme) => {
        if (theme === 'dark') {
            htmlElement.classList.add('dark-mode');
            currentTheme = 'dark';
        } else {
            htmlElement.classList.remove('dark-mode');
            currentTheme = 'light';
        }
        localStorage.setItem('theme', currentTheme);
    };

    const loadThemePreference = () => {
        const savedTheme = localStorage.getItem('theme') || 'light'; // Default to light
        applyTheme(savedTheme);
    };

    const handleThemeToggle = () => {
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(newTheme);
    };

    const initializeApp = async () => {
        loadThemePreference();

        personaList.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Loading personas...</div>';
        try {
            const journalists = await fetchJournalists();
            mockJournalists = journalists;
            renderPersonas(journalists);
        } catch (error) {
            console.error('Failed to fetch journalists:', error);
            personaList.innerHTML = '<p class="status-message error">Failed to load AI personas. Please try again later.</p>';
        }

        sendPitchBtn.addEventListener('click', handleSendPitch);
        themeToggleBtn.addEventListener('click', handleThemeToggle);
    };

    initializeApp();
});