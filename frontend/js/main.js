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
    let journalistPanel = [];

    const BASE_PATH = window.location.pathname

    const fetchJournalists = async () => {
        try {
            const response = await fetch(`${BASE_PATH}journalists`);
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

    const generateResponse = async (pitch_text, journalist_ids) => {
        if (!pitch_text || journalist_ids.length === 0) {
            reject("Please provide pitch text and select at least one persona.");
            return;
        }

        try {
            const reqData = {
                journalist_ids,
                pitch_text
            };

            const reqConfig = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reqData),
            };

            const response = await fetch(`${BASE_PATH}generate_response`, reqConfig);
            if (!response.ok) {
                if (response.status === 404) {
                    console.log("Error occured")
                    throw new Error("Resource doesn't exist.")
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            let responses = [];
            Object.keys(data).forEach((id) => {
                const persona = journalistPanel.find(j => j.id == id);
                if (persona) {
                    let message = data[id];
                    if (message.status === "success") {
                        responses.push(`**Response from ${persona.name} (${persona.role}):**\n${message.response}`);
                    } else {
                        responses.push(`**Response from ${persona.name} (${persona.role}):**\n-- No message --`);
                    }
                } else {
                    responses.push(`Error: Persona not found.`)
                }
            });
            return responses;
        } catch (error) {
            throw new Error("An error occured : ".error)
        }
    };

    const renderResponses = (responses) => {
        responsesContainer.innerHTML = '';
        console.log(responses)
        if (responses.length === 0) {
            responsesContainer.innerHTML = '<p class="status-message no-responses">No responses generated or available.</p>';
            return;
        }
        responses.forEach(response => {
            const responseCard = document.createElement('div');
            responseCard.className = 'response-card';
            const formattedResponse = response.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            responseCard.innerHTML = `<p>${formattedResponse.replace(/\n/g, '<br>')}</p>`;
            responsesContainer.appendChild(responseCard);
        });
    };

    function displayNlpResults(data) {
        console.log("yo")
        const pitchNlpDiv = document.querySelector(".pitch-nlp-components");
        pitchNlpDiv.innerHTML = '';

        function normalizeKey(key) {
            return key
                .replace(/_/g, ' ')
                .replace(/\b\w/g, char => char.toUpperCase());
        }

        function createListItems(items, container) {
            if (!Array.isArray(items)) {
                const span = document.createElement('span');
                span.className = 'list-item';
                span.textContent = items;
                container.appendChild(span);
                return;
            }

            items.forEach(item => {
                if (Array.isArray(item)) {
                    if (item.length === 2 && typeof item[0] === 'string' && typeof item[1] === 'string') {
                        const span = document.createElement('span');
                        span.className = 'list-item';
                        span.textContent = `${item[0]} (${item[1]})`;
                        container.appendChild(span);
                    } else {
                        createListItems(item, container);
                    }
                } else {
                    const span = document.createElement('span');
                    span.className = 'list-item';
                    span.textContent = item;
                    container.appendChild(span);
                }
            });
        }

        for (const key in data) {
            if (Object.prototype.hasOwnProperty.call(data, key)) {
                const value = data[key];
                const normalizedTitle = normalizeKey(key);

                const sectionTitle = document.createElement('h3');
                sectionTitle.className = 'section-title !text-base mb-2';
                sectionTitle.textContent = `${normalizedTitle}:`;
                pitchNlpDiv.appendChild(sectionTitle);

                const contentContainer = document.createElement('div');
                contentContainer.className = 'flex flex-wrap gap-2 mb-4';

                if (value && (Array.isArray(value) && value.length > 0 || typeof value === 'string' && value.length > 0)) {
                    createListItems(value, contentContainer);
                    pitchNlpDiv.appendChild(contentContainer);
                } else {
                    const p = document.createElement('p');
                    p.className = 'text-gray-500 text-sm mb-4';
                    p.textContent = `No ${normalizedTitle.toLowerCase()} found.`;
                    pitchNlpDiv.appendChild(p);
                }
            }
        }
    }

    function displayNlpResultss(data) {
        pitchNlpDiv = document.querySelector(".pitch-nlp-components")
        pitchNlpDiv.innerHTML = '';

        if (data.entities && data.entities.length > 0) {
            const entitiesTitle = document.createElement('h3');
            entitiesTitle.className = 'section-title !text-base mb-2';
            entitiesTitle.textContent = 'Entities:';
            pitchNlpDiv.appendChild(entitiesTitle);

            const entitiesContainer = document.createElement('div');
            entitiesContainer.className = 'flex flex-wrap gap-2 mb-4';
            data.entities.forEach(entity => {
                const span = document.createElement('span');
                span.className = 'list-item entity-item';
                span.textContent = `${entity[0]} (${entity[1]})`;
                entitiesContainer.appendChild(span);
            });
            pitchNlpDiv.appendChild(entitiesContainer);
        } else {
            const p = document.createElement('p');
            p.className = 'text-gray-500 text-sm';
            p.textContent = 'No entities found.';
            pitchNlpDiv.appendChild(p);
        }

        if (data.noun_chunks && data.noun_chunks.length > 0) {
            const chunksTitle = document.createElement('h3');
            chunksTitle.className = 'section-title !text-base mb-2';
            chunksTitle.textContent = 'Noun Chunks:';
            pitchNlpDiv.appendChild(chunksTitle);

            const chunksContainer = document.createElement('div');
            chunksContainer.className = 'flex flex-wrap gap-2 mb-2';
            data.noun_chunks.forEach(chunk => {
                const span = document.createElement('span');
                span.className = 'list-item chunk-item';
                span.textContent = chunk;
                chunksContainer.appendChild(span);
            });
            pitchNlpDiv.appendChild(chunksContainer);
        } else {
            const p = document.createElement('p');
            p.className = 'text-gray-500 text-sm';
            p.textContent = 'No noun chunks found.';
            pitchNlpDiv.appendChild(p);
        }

        if (data.summary && data.summary.length > 0) {
            const chunksTitle = document.createElement('h3');
            chunksTitle.className = 'section-title !text-base mb-4';
            chunksTitle.textContent = 'Summary :';
            pitchNlpDiv.appendChild(chunksTitle);

            const chunksContainer = document.createElement('div');
            chunksContainer.className = 'flex flex-wrap gap-2';

            const span = document.createElement('span');
            span.textContent = data.summary;
            span.className = 'chunk-item';
            chunksContainer.appendChild(span);
            pitchNlpDiv.appendChild(chunksContainer);
        } else {
            const p = document.createElement('p');
            p.className = 'text-gray-500 text-sm';
            p.textContent = 'No summary found.';
            pitchNlpDiv.appendChild(p);
        }
    }

    const getPitchAnalysis = async (pitch_text, journalist_ids) => {
        try {
            const reqData = {
                journalist_ids,
                pitch_text
            };

            const reqConfig = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reqData),
            };

            const response = await fetch(`${BASE_PATH}analyze_pitch`, reqConfig);

            if (!response.ok) {
                const errorData = await response.json().catch(() => response.text());
                console.error(`HTTP error! status: ${response.status}`, errorData);
                throw new Error(`Server responded with status ${response.status}: ${JSON.stringify(errorData)}`);
            }

            const responseBody = await response.json();

            return responseBody;

        } catch (error) {
            console.error("Error during pitch analysis fetch:", error);
            return { error: error.message || "Failed to get pitch analysis." };
        }
    }

    const handleSendPitch = async () => {
        const pitchText = pitchTextarea.value.trim();
        const personaIdsArray = Array.from(selectedPersonaIds);

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

            const pitchAnalysis = await getPitchAnalysis(pitchText, personaIdsArray);
            displayNlpResults(pitchAnalysis);
            console.log(pitchAnalysis);
        } catch (error) {
            console.error('Error generating responses:', error);
            responsesContainer.innerHTML = `<p class="status-message error">Error: ${error}. Please try again.</p>`;
            pitchStatusMessage.textContent = 'Failed to generate responses.';
            pitchStatusMessage.classList.remove('loading', 'success');
            pitchStatusMessage.classList.add('error');
        } finally {
            sendPitchBtn.disabled = false;
            sendPitchBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Get reactions and insight';

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
            journalistPanel = journalists;
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

const PressReleaseModule = (function () {
    const releases = [
        {
            title: "NeuronForge Launches AI-Powered Tool to Help Students Learn Faster and Smarter",
            body: `FOR IMMEDIATE RELEASE

NeuronForge Launches AI-Powered Tool to Help Students Learn Faster and Smarter

San Francisco, CA – June 19, 2025 – NeuronForge, a Silicon Valley-based AI startup, today announced the launch of its flagship product, StudyPilot, an adaptive learning assistant designed to personalize education through advanced generative AI.

StudyPilot uses real-time assessments, GPT-powered tutoring, and multimodal feedback to help high school and college students grasp complex subjects faster. Built on OpenAI’s GPT-4.5 model and fine-tuned with pedagogical research, the tool dynamically adapts to individual learning patterns.

“We believe learning should be personal, intuitive, and scalable. StudyPilot is like having a private tutor for every student,” said Priya Raman, CEO of NeuronForge.

StudyPilot is now available for institutions and individual students via subscription, with discounted access for Title I schools.

For more information, visit: www.neuronforge.ai`
        },
        {
            title: "New Alzheimer’s Drug Shows Promising Results in Phase II Clinical Trial",
            body: `FOR IMMEDIATE RELEASE

New Alzheimer’s Drug Shows Promising Results in Phase II Clinical Trial

Boston, MA – June 19, 2025 – BioNerva Therapeutics announced today that its investigational drug, BN-221, demonstrated significant cognitive improvement in early Alzheimer’s patients during its Phase II clinical trial.

Conducted across 8 U.S. research centers, the trial showed a 27% improvement in memory retention scores compared to the placebo group. BN-221 is a novel small molecule targeting tau protein aggregation, a key hallmark of Alzheimer’s.

“These results are encouraging and could represent a breakthrough for millions of patients,” said Dr. Elena Martinez, Chief Scientific Officer at BioNerva.

The company plans to begin Phase III trials by Q1 2026, pending FDA guidance.`
        },
        {
            title: "Award-Winning Director Arjun Mehta Unveils “The Last Ember” at Cannes",
            body: `FOR IMMEDIATE RELEASE

Award-Winning Director Arjun Mehta Unveils “The Last Ember” at Cannes

Mumbai, India – June 19, 2025 – Acclaimed filmmaker Arjun Mehta debuted his latest psychological thriller, The Last Ember, at the 78th Cannes Film Festival today to a standing ovation.

Starring Tara Rao and Julian Marks, the film follows a war correspondent unraveling a mystery in a post-conflict Eastern European town. Shot across Romania, the UK, and Ladakh, The Last Ember blends geopolitical drama with haunting visuals.

“This is my most personal film yet—about truth, trauma, and redemption,” said Mehta at the press conference.

The Last Ember will release in India, Europe, and North America on August 2, 2025.`
        },
        {
            title: "Tata Textiles Commits to Net Zero by 2040 with Major Green Transition Plan",
            body: `FOR IMMEDIATE RELEASE

Tata Textiles Commits to Net Zero by 2040 with Major Green Transition Plan

New Delhi, India – June 19, 2025 – Tata Textiles, one of India's largest garment exporters, has unveiled a roadmap to achieve net-zero carbon emissions across all operations by 2040.

The plan includes transitioning 70% of energy use to solar and wind by 2030, implementing AI-driven water recycling systems, and eliminating plastic packaging by 2026.

“We want to lead not just in fashion, but in responsible manufacturing,” said Kavita Deshmukh, MD of Tata Textiles.

The initiative, titled Project Revive, is expected to cut over 1 million tons of CO₂ annually and create 12,000 green jobs.`
        },
        {
            title: "Ministry of Urban Affairs Announces ₹6,000 Cr Smart City Revamp Plan",
            body: `FOR IMMEDIATE RELEASE

Ministry of Urban Affairs Announces ₹6,000 Cr Smart City Revamp Plan

New Delhi – June 19, 2025 – The Government of India today announced a ₹6,000 crore investment to upgrade digital and physical infrastructure across 35 Tier-2 cities as part of the next phase of the Smart Cities Mission.

Key initiatives include AI-based traffic control systems, public Wi-Fi in commercial zones, electric bus corridors, and integrated waste management solutions.

“India’s cities are growing fast—this plan ensures they grow smart and sustainably,” said Union Minister for Urban Affairs, Anjali Sinha.

The revamped mission is expected to roll out in phases from September 2025.`
        }
    ];

    function createPopup() {
        const overlay = document.createElement("div");
        overlay.id = "press-release-popup";
        overlay.style.cssText = `
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.6);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
    `;

        const popup = document.createElement("div");
        popup.style.cssText = `
      background: #fff;
      padding: 20px;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      border-radius: 10px;
      box-shadow: 0 0 20px rgba(0,0,0,0.2);
    `;

        const title = document.createElement("h2");
        title.innerText = "Select a Sample Press Release";
        popup.appendChild(title);

        const list = document.createElement("ul");
        list.style.cssText = "list-style:none; padding:0;";

        releases.forEach((release, index) => {
            const li = document.createElement("li");
            const btn = document.createElement("button");
            btn.innerText = release.title;
            btn.style.cssText = `
        display: block;
        width: 100%;
        margin: 10px 0;
        padding: 10px;
        border: none;
        background: #f2f2f2;
        cursor: pointer;
        text-align: left;
        font-size: 16px;
        border-radius: 5px;
      `;
            btn.onclick = () => {
                document.getElementById("pitch-text").value = release.body;
                document.body.removeChild(overlay);
            };
            li.appendChild(btn);
            list.appendChild(li);
        });

        popup.appendChild(list);
        overlay.appendChild(popup);
        document.body.appendChild(overlay);
    }

    function bindEvents() {
        const trigger = document.getElementById("samplePressReleaseList");
        if (trigger) {
            trigger.addEventListener("click", createPopup);
        }
    }

    return {
        init: bindEvents
    };
})();

document.addEventListener("DOMContentLoaded", function () {
    PressReleaseModule.init();

    document.querySelector("#showDebugBtn").addEventListener("click", e => {
        let debugContainer = document.querySelector('.pitch-nlp');
        if (debugContainer.classList.contains("active")) {
            debugContainer.classList.remove("active")
        } else {
            debugContainer.classList.add("active")
        }
    })
});