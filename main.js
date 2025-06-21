// File: main.js
const { app, BrowserWindow, shell, ipcMain } = require('electron');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

let mainWindow;
const GRADIO_PORT = 7860; // Keep this, app.py still uses it
const GRADIO_URL = `http://127.0.0.1:${GRADIO_PORT}`;

const PYTHON_VENV_NAME = 'odin311'; // Your venv name

// Child processes
let gradioProcess = null;
let browserUseProcess = null;

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
        },
    });
    mainWindow.loadFile('index.html');
    // mainWindow.webContents.openDevTools(); // For debugging Electron UI

    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
    mainWindow.on('closed', () => mainWindow = null);
}

function startPythonProcess(scriptName, processLogPrefix, port = null) {
    const pythonSrcDir = app.isPackaged
        ? path.join(process.resourcesPath, 'app', 'python_src')
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

    const args = ['-u', scriptPath];
    const env = {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "UTF-8",
    };
    const child = spawn(pythonExecutable, args, {
        cwd: pythonSrcDir,
        env,
        shell: false,
    });

    child.stdout.on('data', (data) => {
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('pty-data', `[${processLogPrefix}] ${data.toString()}`);
        }
    });
    child.stderr.on('data', (data) => {
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('pty-data', `[${processLogPrefix}-ERR] ${data.toString()}`);
        }
    });
    child.on('exit', (code, signal) => {
        const exitMsg = `[${processLogPrefix}] Process exited (code: ${code}, signal: ${signal})\r\n`;
        console.log(exitMsg);
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('pty-data', exitMsg);
            if (processLogPrefix === "GRADIO") {
                mainWindow.webContents.send('backend-died', { code, signal });
            }
        }
    });
    console.log(`[${processLogPrefix}] Python process '${scriptName}' initiated.`);
    return child;
}

app.on('ready', () => {
    console.log("Electron app ready.");
    createWindow();

    // Start the mcp_browseruse.py server
    browserUseProcess = startPythonProcess('mcp_browseruse.py', "BROWSER_SSE");

    // Start the main Gradio app.py server
    gradioProcess = startPythonProcess('app.py', "GRADIO", GRADIO_PORT);

    // Give Python server some time to start before attempting to load Gradio in iframe.
    setTimeout(() => {
        if (mainWindow && gradioProcess) {
            console.log(`Attempting to trigger Gradio page load: ${GRADIO_URL}`);
            mainWindow.webContents.send('load-gradio', GRADIO_URL);
        } else if (mainWindow && !gradioProcess) {
            const msg = "Gradio backend process failed to start. Check logs in main terminal and Electron UI.\r\n";
            console.error(msg);
            if(mainWindow.webContents) mainWindow.webContents.send('pty-data', `[ELECTRON-ERROR] ${msg}`);
        }
    }, 7000);
});

function killProcess(childProc, name) {
    if (childProc) {
        console.log(`Terminating ${name} process (PID: ${childProc.pid})...`);
        try {
            childProc.kill('SIGKILL');
            console.log(`${name} process kill signal sent.`);
        } catch (e) {
            console.error(`Error killing ${name} process: ${e.message}`);
        }
    }
    return null;
}

app.on('will-quit', () => {
    console.log('Terminating backend processes...');
    if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('pty-data', '[ELECTRON] Exiting: Terminating backend processes...\r\n');
    }
    gradioProcess = killProcess(gradioProcess, "Gradio");
    browserUseProcess = killProcess(browserUseProcess, "BrowserUse SSE");
});

app.on('window-all-closed', () => (process.platform !== 'darwin') && app.quit());
app.on('activate', () => (mainWindow === null) && createWindow());