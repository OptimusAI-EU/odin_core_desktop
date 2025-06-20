// File: main.js
const { app, BrowserWindow, shell, ipcMain } = require('electron');
const path = require('path');
const os = require('os');
const pty = require('node-pty');

let mainWindow;
const GRADIO_PORT = 7860; // Keep this, app.py still uses it
const GRADIO_URL = `http://127.0.0.1:${GRADIO_PORT}`;

const PYTHON_VENV_NAME = 'odin'; // Your venv name

// Child processes
let gradioPtyProcess = null;
let browserUsePtyProcess = null;

// Helper to determine Python executable path within the venv
function getPythonExecutablePath(basePythonSrcDir) {
    if (os.platform() === 'win32') {
        return path.join(basePythonSrcDir, PYTHON_VENV_NAME, 'Scripts', 'python.exe');
    }
    return path.join(basePythonSrcDir, PYTHON_VENV_NAME, 'bin', 'python');
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 950, // Adjusted height for terminal
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false, // Important for security
            // nodeIntegrationInWorker: false, // If you use workers
            // webviewTag: false, // If you were using webview
        },
    });
    mainWindow.loadFile('index.html');
    // mainWindow.webContents.openDevTools(); // For debugging Electron UI

    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
    mainWindow.on('closed', () => mainWindow = null);

    // Relay terminal resize events
    ipcMain.on('terminal-resize', (event, { cols, rows }) => {
        if (gradioPtyProcess) {
            try {
                gradioPtyProcess.resize(cols, rows);
            } catch (e) {
                console.warn("Error resizing Gradio PTY:", e.message);
            }
        }
        // You might want a separate terminal or combined output for browserUsePtyProcess
        // If combined, this resize applies to the shared terminal.
        // If separate, you'd need to manage its resize too.
    });

    // Relay input to the Gradio PTY (e.g., for Ctrl+C)
    ipcMain.on('terminal-input', (event, data) => {
        if (gradioPtyProcess) {
            gradioPtyProcess.write(data);
        }
    });
}

function startPythonProcess(scriptName, processLogPrefix, port = null) {
    const pythonSrcDir = app.isPackaged
        ? path.join(process.resourcesPath, 'app', 'python_src') // Adjusted for "app/python_src"
        : path.join(__dirname, 'python_src');

    const pythonExecutable = getPythonExecutablePath(pythonSrcDir);
    const scriptPath = path.join(pythonSrcDir, scriptName);

    const fs = require('fs');
    if (!fs.existsSync(pythonExecutable)) {
        const errorMsg = `Python executable not found for venv '${PYTHON_VENV_NAME}': ${pythonExecutable}`;
        console.error(errorMsg);
        if (mainWindow && mainWindow.webContents) {
            mainWindow.webContents.send('pty-data', `[${processLogPrefix}-ERROR] ${errorMsg}\r\n`);
        }
        return null;
    }
    if (!fs.existsSync(scriptPath)) {
        const errorMsg = `Python script not found: ${scriptPath}`;
        console.error(errorMsg);
        if (mainWindow && mainWindow.webContents) {
            mainWindow.webContents.send('pty-data', `[${processLogPrefix}-ERROR] ${errorMsg}\r\n`);
        }
        return null;
    }

    const shell = os.platform() === 'win32' ? 'powershell.exe' : 'bash'; // Or 'cmd.exe'
    const ptyProcess = pty.spawn(pythonExecutable, ['-u', scriptPath], { // -u for unbuffered
        name: 'xterm-color',
        cols: 80, // Initial columns
        rows: 30, // Initial rows
        cwd: pythonSrcDir, // Run script from its directory
        env: {
            ...process.env,
            PYTHONUNBUFFERED: "1",
            PYTHONIOENCODING: "UTF-8",
            // GRADIO_SERVER_PORT: port ? port.toString() : undefined // app.py handles its own port
        },
        // useConpty: os.platform() === 'win32' // Use ConPTY on Windows if available (default)
    });

    ptyProcess.onData((data) => {
        // process.stdout.write(`[${processLogPrefix}] ${data}`); // Log to main Electron console
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('pty-data', `[${processLogPrefix}] ${data}`);
        }
    });

    ptyProcess.onExit(({ exitCode, signal }) => {
        const exitMsg = `[${processLogPrefix}] Process exited (code: ${exitCode}, signal: ${signal})\r\n`;
        console.log(exitMsg);
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('pty-data', exitMsg);
            if (processLogPrefix === "GRADIO") { // If the main Gradio server dies
                 mainWindow.webContents.send('backend-died', {code: exitCode, signal});
            }
        }
    });
    console.log(`[${processLogPrefix}] Python process '${scriptName}' initiated with PTY.`);
    return ptyProcess;
}

app.on('ready', () => {
    console.log("Electron app ready.");
    createWindow(); // Create window first so it can receive logs

    // Start the mcp_browseruse.py server
    browserUsePtyProcess = startPythonProcess('mcp_browseruse.py', "BROWSER_SSE");

    // Start the main Gradio app.py server
    gradioPtyProcess = startPythonProcess('app.py', "GRADIO", GRADIO_PORT);


    // Give Python server some time to start before attempting to load Gradio in iframe.
    setTimeout(() => {
        if (mainWindow && gradioPtyProcess) {
            console.log(`Attempting to trigger Gradio page load: ${GRADIO_URL}`);
            mainWindow.webContents.send('load-gradio', GRADIO_URL);
        } else if (mainWindow && !gradioPtyProcess) {
            const msg = "Gradio backend process failed to start. Check logs in main terminal and Electron UI.\r\n";
            console.error(msg);
            if(mainWindow.webContents) mainWindow.webContents.send('pty-data', `[ELECTRON-ERROR] ${msg}`);
        }
    }, 7000); // Increased delay for backend to start, especially with PTY
});

function killPtyProcess(ptyProc, name) {
    if (ptyProc) {
        console.log(`Terminating ${name} PTY process (PID: ${ptyProc.pid})...`);
        try {
            // Sending SIGKILL directly. For more grace, try 'SIGINT' or ptyProc.write('\x03') first.
            ptyProc.kill('SIGKILL'); // Or a more graceful signal like 'SIGTERM'
            console.log(`${name} PTY process kill signal sent.`);
        } catch (e) {
            console.error(`Error killing ${name} PTY process: ${e.message}`);
        }
    }
    return null;
}

app.on('will-quit', () => {
    console.log('Terminating backend processes...');
    if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('pty-data', '[ELECTRON] Exiting: Terminating backend processes...\r\n');
    }
    gradioPtyProcess = killPtyProcess(gradioPtyProcess, "Gradio");
    browserUsePtyProcess = killPtyProcess(browserUsePtyProcess, "BrowserUse SSE");
});

app.on('window-all-closed', () => (process.platform !== 'darwin') && app.quit());
app.on('activate', () => (mainWindow === null) && createWindow());