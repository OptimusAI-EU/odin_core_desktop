// File: preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    onLoadGradio: (callback) => ipcRenderer.on('load-gradio', (_event, url) => callback(url)),
    onTerminalLog: (callback) => { // Renamed for clarity
        const handler = (_event, data) => callback(data);
        ipcRenderer.on('terminal-log', handler);
        return () => ipcRenderer.removeListener('terminal-log', handler); // Cleanup
    },
    onBackendDied: (callback) => ipcRenderer.on('backend-died', (_event, data) => callback(data))
});

console.log("Preload script (simplified log version) executed.");