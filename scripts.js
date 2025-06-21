// File: scripts.js

document.addEventListener('DOMContentLoaded', () => {
    const iframe = document.getElementById('gradio-iframe');
    const loadingMessage = document.getElementById('loading-message');
    const terminalContainer = document.getElementById('terminal-container');
    let term;

    console.log('[scripts.js] DOMContentLoaded. Initializing application logic...');

    // ---- Get Constructors from Global Scope (set by UMD <script src> tags) ----
    const TerminalConstructor = window.Terminal;
    const FitAddonClassFromGlobalObject = window.FitAddon && typeof window.FitAddon.FitAddon === 'function'
        ? window.FitAddon.FitAddon
        : null;

    console.log('[scripts.js] typeof TerminalConstructor:', typeof TerminalConstructor);
    console.log('[scripts.js] typeof FitAddonClassFromGlobalObject:', typeof FitAddonClassFromGlobalObject);

    // ---- SCRIPT EXECUTION GUARD ----
    if (typeof TerminalConstructor !== 'function' || typeof FitAddonClassFromGlobalObject !== 'function') {
        const errorMsg = 'FATAL: Xterm.js window.Terminal or window.FitAddon.FitAddon not available correctly. Check <script src> paths in index.html and UMD bundle integrity. UI cannot proceed.';
        console.error(errorMsg, 'window.Terminal was:', TerminalConstructor, 'window.FitAddon was:', window.FitAddon, 'FitAddon.FitAddon was:', FitAddonClassFromGlobalObject);
        const body = document.body;
        if (body) {
           body.innerHTML = `<div class="fatal-error-display">${errorMsg}<br/>Check Electron DevTools Console.</div>`;
        }
        return; // Stop further script execution
    }

    // --- XTERM.JS INITIALIZATION ---
    try {
        console.log('[scripts.js] Initializing Terminal instance...');
        term = new TerminalConstructor({ // Use the found constructor
            cursorBlink: true,
            convertEol: true,
            fontSize: 13,
            fontFamily: 'Consolas, "Courier New", monospace',
            theme: {
                background: '#1e1e1e',
                foreground: '#d4d4d4',
                cursor: '#d4d4d4',
                selectionBackground: '#555555',
            }
        });

        console.log('[scripts.js] Initializing FitAddon instance...');
        const fitAddonInstance = new FitAddonClassFromGlobalObject(); // Use the found constructor
        term.loadAddon(fitAddonInstance);
        
        console.log('[scripts.js] Opening terminal in container...');
        term.open(terminalContainer);
        
        console.log('[scripts.js] Performing initial fit...');
        try {
            fitAddonInstance.fit();
        } catch (e) {
            console.warn("[scripts.js] Initial fitAddonInstance.fit() failed:", e);
        }
        console.log('[scripts.js] Xterm.js setup complete.');

        // Handle resize (copy this from your previous full index.html script block)
        const resizeObserver = new ResizeObserver(() => {
            try {
                fitAddonInstance.fit();
                if (window.electronAPI && window.electronAPI.sendTerminalResize && term) {
                    window.electronAPI.sendTerminalResize(term.cols, term.rows);
                }
            } catch (e) { /* ignore */ }
        });
        resizeObserver.observe(terminalContainer);
        
        window.addEventListener('resize', () => {
            try { fitAddonInstance.fit(); } catch(e) { /* ignore */ }
        });

    } catch (e_xterm_init) {
        const xtermErrorMsg = `FATAL: Error initializing Xterm.js (UMD from scripts.js): ${e_xterm_init.message}.`;
        console.error(xtermErrorMsg, e_xterm_init);
        document.body.innerHTML = `<div class="fatal-error-display">${xtermErrorMsg}</div>`;
        return; // Stop script
    }

    // --- ELECTRON API INTERACTION & GRADIO LOADING ---
    // (This entire section should be identical to what you had in your complete index.html's script block
    //  that was working with the diagnostics, starting from `if (window.electronAPI) { ... }`)
    if (window.electronAPI) {
        term.writeln('\x1b[1;36mWelcome to ODIN Core Terminal (UMD Loaded)\x1b[0m\r\n');
        term.writeln('Backend process logs will appear here...\r\n');

        if (window.electronAPI.onPtyData) {
            window.electronAPI.onPtyData((data) => {
                if (term) term.write(data);
            });
        } else { console.warn("[scripts.js] window.electronAPI.onPtyData is not available!"); }

        if (term && window.electronAPI.sendPtyInput) {
            term.onData(data => {
                window.electronAPI.sendPtyInput(data);
            });
        } else { console.warn("[scripts.js] term or window.electronAPI.sendPtyInput is not available for term.onData!");}

        if (window.electronAPI.onBackendDied) {
            window.electronAPI.onBackendDied((exitInfo) => {
                if (term) {
                    term.writeln(`\r\n\x1b[1;31m--- GRADIO BACKEND EXITED (Code: ${exitInfo.code}, Signal: ${exitInfo.signal}) ---\x1b[0m\r\n`);
                    term.writeln('\x1b[33mGradio interface may be unresponsive. Check logs.\x1b[0m\r\n');
                }
                if (iframe) iframe.src = 'about:blank';
                if (loadingMessage) {
                    loadingMessage.textContent = 'Backend process terminated. Please restart or check logs.';
                    loadingMessage.style.display = 'block';
                }
            });
        } else { console.warn("[scripts.js] window.electronAPI.onBackendDied is not available!"); }

        if (window.electronAPI.onLoadGradio) {
            window.electronAPI.onLoadGradio((gradioUrl) => {
                if (term) term.writeln(`\x1b[36mRenderer: Received Gradio URL (${gradioUrl}). Attempting load...\x1b[0m\r\n`);
                
                let gradioLoadAttempts = 0;
                const maxGradioLoadAttempts = 30; 
                const gradioRetryDelay = 3000;

                function attemptLoadGradio() { // Make sure this function is defined here
                    if (gradioLoadAttempts >= maxGradioLoadAttempts) {
                        const failMsg = `Failed to load Gradio from ${gradioUrl} after ${maxGradioLoadAttempts} attempts. Check Python server logs and Electron DevTools console.`;
                        if (loadingMessage) loadingMessage.innerHTML = failMsg.replace(/\n/g, '<br>');
                        if (term) term.writeln(`\x1b[1;31m${failMsg}\x1b[0m\r\n`);
                        return;
                    }
                    gradioLoadAttempts++;
                    if (loadingMessage) loadingMessage.textContent = `Attempting Gradio load (${gradioLoadAttempts}/${maxGradioLoadAttempts})...`;
                    if (term) term.writeln(`\x1b[36mRenderer: Attempting Gradio load ${gradioLoadAttempts}/${maxGradioLoadAttempts} from ${gradioUrl}\x1b[0m\r\n`);

                    fetch(gradioUrl, { method: 'HEAD', mode: 'no-cors', cache: 'no-cache' })
                        .then(response => {
                            if (term) term.writeln('\x1b[32mRenderer: Gradio server appears responsive. Setting iframe source.\x1b[0m\r\n');
                            if (iframe) iframe.src = gradioUrl;
                        })
                        .catch(error => {
                            if (term) term.writeln(`\x1b[33mRenderer (Attempt ${gradioLoadAttempts}): Gradio connection to ${gradioUrl} failed: ${error.message}\x1b[0m\r\n`);
                            if (gradioLoadAttempts < maxGradioLoadAttempts) {
                                setTimeout(attemptLoadGradio, gradioRetryDelay);
                            } else {
                                const failMsg = `Failed to load Gradio from ${gradioUrl} after ${maxGradioLoadAttempts} attempts. Max retries reached.`;
                                if (loadingMessage) loadingMessage.innerHTML = failMsg.replace(/\n/g, '<br>');
                                if (term) term.writeln(`\x1b[1;31m${failMsg}\x1b[0m\r\n`);
                            }
                        });
                }

                if (iframe) {
                    iframe.onload = () => {
                        if (term) term.writeln('\x1b[32mRenderer: Gradio iframe content loaded successfully.\x1b[0m\r\n');
                        if (loadingMessage) loadingMessage.style.display = 'none';
                    };
                    iframe.onerror = (e) => {
                        if (term) term.writeln(`\x1b[1;31mRenderer: Gradio iframe reported an error (type: ${e ? e.type : 'unknown'}).\x1b[0m\r\n`);
                    };
                }
                attemptLoadGradio(); // Call the defined function
            });
        } else { console.warn("[scripts.js] window.electronAPI.onLoadGradio is not available!"); }

        if (term) term.writeln("\x1b[36mRenderer: UI Initialized (UMDs). electronAPI good. Waiting for events...\x1b[0m\r\n");

    } else {
        const apiError = "CRITICAL ERROR: window.electronAPI not available (scripts.js). Preload script problem. Terminal and Gradio will not function.";
        console.error(apiError);
        if (loadingMessage) loadingMessage.textContent = apiError;
        const termContainerFallback = document.getElementById('terminal-container');
        if (termContainerFallback) termContainerFallback.innerHTML = `<div class="fatal-error-display">${apiError}</div>`;
    }
}); // End of DOMContentLoaded listener