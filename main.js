// File: main.js
const { app, BrowserWindow, shell, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process'); // Using spawn directly
const os = require('os');

let mainWindow;
let pythonProcess = null;
const GRADIO_PORT = 7860;
const GRADIO_URL = `http://127.0.0.1:${GRADIO_PORT}`;

const CONDA_ENV_NAME = 'odin';
const CONDA_EXECUTABLE = 'conda'; // Assumes 'conda' is in system PATH

// Helper to send logs to renderer
function sendLogToRenderer(type, message) {
    if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('terminal-log', { type, message });
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 950,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
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

function startPythonBackend() {
    const pythonAppDir = app.isPackaged
        ? path.join(process.resourcesPath, 'python_src')
        : path.join(__dirname, 'python_src');
    const scriptRelativePath = 'app.py';

    const logAndSend = (type, msg) => {
        const fullMsg = `[Backend-${type.toUpperCase()}]: ${msg}`;
        console.log(fullMsg); // Log to main Electron console
        sendLogToRenderer(type, msg); // Send original message to renderer
    };

    logAndSend('status', `Python app dir: ${pythonAppDir}, script: ${scriptRelativePath}`);
    logAndSend('status', `Using Conda env: ${CONDA_ENV_NAME} via ${CONDA_EXECUTABLE}`);

    const command = CONDA_EXECUTABLE;
    const args = [
        'run',
        '-n', CONDA_ENV_NAME,
        '--no-capture-output', // Still good for conda to not buffer excessively
        'python',
        '-u', // CRITICAL: Unbuffered Python output for stdout/stderr
        scriptRelativePath
    ];

    const options = {
        cwd: pythonAppDir,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: "1", // Double ensure unbuffered for Python
            PYTHONIOENCODING: "UTF-8", // Ensure UTF-8 for output
            // GRADIO_SERVER_NAME: "127.0.0.1", // Optional
            // GRADIO_SERVER_PORT: GRADIO_PORT.toString(),
        },
        // Process group handling for termination:
        detached: os.platform() !== 'win32', // On Unix, create a new process group
    };

    if (os.platform() === 'win32') {
        options.shell = true; // `conda` is often a .bat on Windows, needing a shell
        // `detached` behaves differently with `shell: true` on Windows.
        // `taskkill` will be our primary tool for Windows termination.
    }

    logAndSend('status', `Spawning: ${command} ${args.join(' ')}`);
    pythonProcess = spawn(command, args, options);

    // --- Stream Handling ---
    // Make sure to handle encoding correctly if output isn't plain ASCII
    pythonProcess.stdout.setEncoding('utf8');
    pythonProcess.stderr.setEncoding('utf8');

    pythonProcess.stdout.on('data', (data) => {
        // data is already a string due to setEncoding
        process.stdout.write(data); // Mirror to Electron's main console (where you ran npm start)
        sendLogToRenderer('stdout', data);
    });

    pythonProcess.stderr.on('data', (data) => {
        process.stderr.write(data); // Mirror to Electron's main console
        sendLogToRenderer('stderr', data);
    });

    pythonProcess.on('error', (err) => {
        logAndSend('error', `Failed to start Python process: ${err.message}`);
        pythonProcess = null;
    });

    pythonProcess.on('close', (code, signal) => {
        logAndSend('status', `Python process exited (code: ${code}, signal: ${signal})`);
        pythonProcess = null;
        if (mainWindow && mainWindow.webContents && !mainWindow.isDestroyed()) {
            if (code !== 0 || signal) { // If exited with error or was signaled
                 mainWindow.webContents.send('backend-died', {code, signal});
            }
        }
    });
    logAndSend('status', 'Python backend process initiated.');
}

app.on('ready', () => {
    console.log("Electron app ready.");
    startPythonBackend();
    createWindow();

    setTimeout(() => {
        if (mainWindow && pythonProcess) {
            console.log(`Attempting to trigger Gradio page load: ${GRADIO_URL}`);
            mainWindow.webContents.send('load-gradio', GRADIO_URL);
        } else if (mainWindow && !pythonProcess) {
            const msg = "Python backend process failed to start. Check logs in main terminal and Electron UI.";
            console.error(msg);
            if(mainWindow.webContents) sendLogToRenderer('error', msg);
        }
    }, 7000); // Delay for backend to start
});

app.on('will-quit', () => {
    if (pythonProcess) {
        console.log('Terminating Python backend process and its children...');
        sendLogToRenderer('status', 'Exiting: Terminating Python backend...');

        const pid = pythonProcess.pid;
        if (os.platform() === 'win32') {
            // On Windows, pythonProcess.pid is the PID of the shell (cmd.exe/powershell.exe)
            // if shell:true was used. taskkill /T /F will kill the shell and its descendants.
            try {
                // spawn is used here to run taskkill asynchronously and not block the quit.
                spawn('taskkill', ['/PID', pid.toString(), '/F', '/T'], { detached: true, stdio: 'ignore' }).unref();
                console.log(`Sent taskkill /F /T to PID ${pid}`);
            } catch (e) {
                console.error("taskkill command failed:", e);
                // Fallback to killing the direct shell process if taskkill spawn fails (unlikely)
                if(pythonProcess && !pythonProcess.killed) pythonProcess.kill('SIGKILL');
            }
        } else {
            // On Unix-like systems (Linux, macOS):
            // If `detached: true` was used, pythonProcess.pid is the process group ID (PGID) leader.
            // Sending a signal to -pid (negative PID) kills the entire process group.
            try {
                process.kill(-pid, 'SIGINT'); // Send SIGINT to the process group
                console.log(`Sent SIGINT to process group -${pid}`);
                // Set a timeout to forcefully kill if SIGINT doesn't work
                setTimeout(() => {
                    if (pythonProcess && !pythonProcess.killed) { // Check if still running
                        console.log(`Process group -${pid} did not exit, sending SIGKILL.`);
                        try { process.kill(-pid, 'SIGKILL'); }
                        catch (e) { console.warn("Failed to SIGKILL process group (already exited?):", e.message); }
                    }
                }, 2000); // 2-second grace period
            } catch (e) {
                console.error(`Failed to send SIGINT to process group -${pid}. Trying direct SIGKILL to ${pid}:`, e.message);
                if(pythonProcess && !pythonProcess.killed) pythonProcess.kill('SIGKILL'); // Fallback
            }
        }
        pythonProcess = null; // Assume it will be killed
    }
});

app.on('window-all-closed', () => (process.platform !== 'darwin') && app.quit());
app.on('activate', () => (mainWindow === null) && createWindow());