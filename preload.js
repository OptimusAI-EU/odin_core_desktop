// File: preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Gradio IFrame related
    onLoadGradio: (callback) => ipcRenderer.on('load-gradio', (_event, url) => callback(url)),
    onBackendDied: (callback) => ipcRenderer.on('backend-died', (_event, data) => callback(data)),

    // PTY / Terminal related
    onPtyData: (callback) => {
        const handler = (_event, data) => callback(data);
        ipcRenderer.on('pty-data', handler);
        return () => ipcRenderer.removeListener('pty-data', handler); // Cleanup
    },
    sendPtyInput: (data) => ipcRenderer.send('terminal-input', data),
    sendTerminalResize: (cols, rows) => ipcRenderer.send('terminal-resize', { cols, rows })
});

console.log("Preload script with PTY support executed.");