// File: preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // For Gradio IFrame loading
    onLoadGradio: (callback) => ipcRenderer.on('load-gradio', (_event, url) => callback(url)),
    onBackendDied: (callback) => ipcRenderer.on('backend-died', (_event, data) => callback(data)),

    // For receiving stdout/stderr from Python processes spawned by main.js
    // main.js needs to send 'pty-data' (even if it's not strictly PTY anymore)
    onPtyData: (callback) => {
        const handler = (_event, data) => callback(data); // data is expected to be a string
        ipcRenderer.on('pty-data', handler);
        return () => ipcRenderer.removeListener('pty-data', handler);
    }
    // Removed terminal-input and terminal-resize as node-pty is removed from main.js for now
});

console.log("Preload script (simplified for basic Gradio loading & logs) executed.");