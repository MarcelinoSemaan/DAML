import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

export function activate(context: vscode.ExtensionContext) {
    const provider = new DamlDashboardProvider(context);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('daml.dashboardView', provider),
        vscode.commands.registerCommand('daml.toggleStatusView', () => provider.toggleExtension()),
        vscode.commands.registerCommand('daml.showMenu', () => provider.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.scanWorkspace', () => provider.scanWorkspace()),
        vscode.commands.registerCommand('daml.scanActiveFile', () => provider.scanActiveFile()),
        vscode.commands.registerCommand('daml.scanPeFile', () => provider.scanPeFilePicker()),
        vscode.commands.registerCommand('daml.scanJsonClipboard', () => provider.scanJsonClipboard())
    );
}

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 0;
    private isEnabled: boolean = true;
    private recentDetections: Array<{ file: string; severity: string; time: string; prob: number }> = [];
    private rollingHistory: number[] = new Array(12).fill(0);
    private apiAvailable: boolean = false;

    constructor(private context: vscode.ExtensionContext) {
        this.rollingHistory = this.context.globalState.get<number[]>('daml.rollingHistory', new Array(12).fill(0));
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'daml.showMenu';
        this.updateStatusBar();
        this.statusBarItem.show();

        this.checkApiHealth();
        setInterval(() => this.checkApiHealth(), 5000);
        setInterval(() => this.shiftRollingWindow(), 300000);
    }

    private async checkApiHealth(): Promise<void> {
        try {
            const health = await this.apiRequest('/health').catch(() => ({ status: 'error' }));
            const wasAvailable = this.apiAvailable;
            this.apiAvailable = health && health.status === 'ok';

            if (this.apiAvailable && !wasAvailable) {
                this.updateWebview();
                this.updateStatusBar();
            } else if (!this.apiAvailable && wasAvailable) {
                this.updateWebview();
                this.updateStatusBar();
            }
        } catch (e: any) {
            if (this.apiAvailable) {
                this.apiAvailable = false;
                this.updateWebview();
                this.updateStatusBar();
            }
        }
    }

    private async apiRequest(endpoint: string, method: 'GET' | 'POST' = 'GET', body?: any): Promise<any> {
        return new Promise((resolve, reject) => {
            const postData = body ? JSON.stringify(body) : '';
            const options = {
                hostname: '127.0.0.1',
                port: 8000,
                path: endpoint,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const req = http.request(options, (res: http.IncomingMessage) => {
                let data = '';
                res.on('data', (chunk: Buffer) => (data += chunk));
                res.on('end', () => {
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        reject(new Error('Invalid JSON response'));
                    }
                });
            });

            req.on('error', (err: Error) => reject(err));
            if (postData) req.write(postData);
            req.end();
        });
    }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'media'))]
        };

        webviewView.webview.onDidReceiveMessage(async (msg) => {
            if (msg.command === 'resolveThreats') this.resolveThreats();
            if (msg.command === 'resetGraph') this.resetGraph();
            if (msg.command === 'scanWorkspace') this.scanWorkspace();
            if (msg.command === 'scanActiveFile') this.scanActiveFile();
            if (msg.command === 'scanPeFile') this.scanPeFilePicker();
        });

        this.updateWebview();
    }

    public toggleExtension() {
        this.isEnabled = !this.isEnabled;
        vscode.window.showInformationMessage(`DAML Protection ${this.isEnabled ? 'Enabled' : 'Disabled'}`);
        this.updateStatusBar();
        this.updateWebview();
    }

    public async showQuickPickMenu() {
        const toggleLabel = this.isEnabled ? '$(circle-slash) Disable' : '$(play) Enable';
        const items = [
            { label: '$(graph) Open Dashboard', cmd: 'daml.dashboardView.focus' },
            { label: '$(shield) Scan PE File...', cmd: 'daml.scanPeFile' },
            { label: '$(file-json) Scan JSON from Clipboard', cmd: 'daml.scanJsonClipboard' },
            { label: '$(shield) Scan Workspace (Build+Scan)', cmd: 'daml.scanWorkspace' },
            { label: '$(file-code) Scan Active File', cmd: 'daml.scanActiveFile' },
            { label: toggleLabel, cmd: 'daml.toggleStatusView' }
        ];

        const selection = await vscode.window.showQuickPick(items, { title: 'DAML Control' });
        if (selection) vscode.commands.executeCommand(selection.cmd);
    }

    public async scanPeFilePicker() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');
        if (!this.apiAvailable) {
            const action = await vscode.window.showWarningMessage(
                'DAML API server not running. Start it with: uvicorn main:app --host 0.0.0.0 --port 8000',
                'Retry',
                'Cancel'
            );
            if (action === 'Retry') {
                await this.checkApiHealth();
                if (this.apiAvailable) this.scanPeFilePicker();
            }
            return;
        }

        const uris = await vscode.window.showOpenDialog({
            canSelectFiles: true,
            canSelectFolders: false,
            canSelectMany: false,
            filters: {
                'PE Files': ['exe', 'dll', 'sys', 'scr', 'ocx'],
                'Source Files': ['cpp', 'c', 'ts', 'js', 'py', 'rs'],
                'Feature JSON': ['json'],
                'All Files': ['*']
            }
        });

        if (!uris || uris.length === 0) return;

        const filePath = uris[0].fsPath;
        const ext = path.extname(filePath).toLowerCase();

        if (['.exe', '.dll', '.sys', '.scr', '.ocx', '.json'].includes(ext)) {
            return this.scanBinary(filePath, true);
        }

        if (['.cpp', '.c', '.ts', '.js', '.py', '.rs'].includes(ext)) {
            return this.buildAndScan(filePath, true);
        }

        vscode.window.showWarningMessage(`Unsupported file type: ${ext}`);
    }

    public async scanWorkspace() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');
        if (!this.apiAvailable) {
            const action = await vscode.window.showWarningMessage(
                'DAML API server not running. Start it with: uvicorn main:app --host 0.0.0.0 --port 8000',
                'Retry',
                'Cancel'
            );
            if (action === 'Retry') {
                await this.checkApiHealth();
                if (this.apiAvailable) this.scanWorkspace();
            }
            return;
        }

        vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: 'DAML Scanning Workspace...', cancellable: true },
            async (progress, token) => {
                const sourceFiles = await vscode.workspace.findFiles('**/*.{ts,js,py,cpp,c}', '**/node_modules/**', 50);

                if (sourceFiles.length === 0) {
                    vscode.window.showInformationMessage('No source files found.');
                    return;
                }

                let scanned = 0;
                const threats: typeof this.recentDetections = [];

                for (const file of sourceFiles) {
                    if (token.isCancellationRequested) break;
                    progress.report({ message: `Processing ${path.basename(file.fsPath)}...`, increment: 100 / sourceFiles.length });

                    try {
                        const result = await this.buildAndScan(file.fsPath, false);
                        if (result && result.is_malicious) {
                            this.recordDetection();
                            threats.push({
                                file: path.basename(file.fsPath),
                                severity: result.confidence === 'high' ? 'critical' : result.confidence,
                                time: 'just now',
                                prob: result.malicious_probability
                            });
                        }
                    } catch (err: any) {
                        console.log(`Skipping ${file.fsPath}: ${err}`);
                    }
                    scanned++;
                }

                this.threatCount += threats.length;
                this.recentDetections = [...threats, ...this.recentDetections].slice(0, 10);
                this.updateStatusBar();
                this.updateWebview();

                if (threats.length > 0) {
                    vscode.window.showWarningMessage(`DAML: Found ${threats.length} potential threats!`);
                } else {
                    vscode.window.showInformationMessage(`DAML: Scanned ${scanned} files. No threats.`);
                }
            }
        );
    }

    public async scanActiveFile() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return this.scanPeFilePicker();
        }

        const sourcePath = editor.document.fileName;
        const ext = path.extname(sourcePath).toLowerCase();

        if (['.exe', '.dll', '.sys', '.scr', '.ocx', '.json'].includes(ext)) {
            return this.scanBinary(sourcePath, true);
        }

        if (!['.ts', '.js', '.py', '.cpp', '.c', '.rs'].includes(ext)) {
            return vscode.window.showWarningMessage(
                `Unsupported: ${ext}. Use .exe, .dll, .json, .js, .ts, .py, .cpp, .c, or run "Scan File..." command`,
                'Scan File...'
            ).then((sel) => {
                if (sel === 'Scan File...') {
                    vscode.commands.executeCommand('daml.scanPeFile');
                }
            });
        }

        await this.buildAndScan(sourcePath, true);
    }

    public async scanJsonClipboard() {
        if (!this.isEnabled) {
            return vscode.window.showWarningMessage('Enable DAML first.');
        }
        if (!this.apiAvailable) {
            const action = await vscode.window.showWarningMessage(
                'DAML API server not running. Start it with: uvicorn main:app --host 0.0.0.0 --port 8000',
                'Retry',
                'Cancel'
            );
            if (action === 'Retry') {
                await this.checkApiHealth();
                if (this.apiAvailable) this.scanJsonClipboard();
            }
            return;
        }

        const text = await vscode.env.clipboard.readText();
        if (!text.trim()) {
            return vscode.window.showWarningMessage('Clipboard is empty. Copy a raw EMBER JSON record first.');
        }

        let record: any;
        try {
            record = JSON.parse(text);
        } catch {
            return vscode.window.showErrorMessage('Clipboard does not contain valid JSON.');
        }

        return vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: 'Scanning JSON from clipboard...' },
            async (progress) => {
                try {
                    progress.report({ message: 'Sending to DAML API...' });
                    const result = await this.apiRequest('/predict_json', 'POST', { record });

                    if (result.is_malicious) {
                        this.threatCount++;
                        this.recordDetection();
                        this.recentDetections.unshift({
                            file: 'clipboard.json',
                            severity: result.confidence === 'high' ? 'critical' : result.confidence,
                            time: 'just now',
                            prob: result.malicious_probability
                        });
                        this.recentDetections = this.recentDetections.slice(0, 10);
                        this.updateStatusBar();
                        this.updateWebview();

                        vscode.window.showWarningMessage(
                            `⚠️ Threat detected in clipboard JSON! (${(result.malicious_probability * 100).toFixed(0)}% confidence)`
                        );
                    } else {
                        vscode.window.showInformationMessage(
                            `✅ Clipboard JSON safe (${(result.malicious_probability * 100).toFixed(1)}% risk)`
                        );
                    }
                    return result;
                } catch (err: any) {
                    vscode.window.showErrorMessage(`Scan failed: ${err.message || err}`);
                    throw err;
                }
            }
        );
    }

    private async buildAndScan(sourcePath: string, showProgress: boolean): Promise<any | null> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath || '';
        const fileName = path.basename(sourcePath);

        let binaryPath = await this.findExistingBinary(sourcePath, workspaceRoot);
        if (binaryPath) {
            return this.scanBinary(binaryPath, showProgress);
        }

        if (!showProgress) return null;

        const action = await vscode.window.showWarningMessage(`No EXE found for ${fileName}. Build now?`, 'Build & Scan', 'Cancel');

        if (action !== 'Build & Scan') return null;

        let builtBinaryPath: string | null = null;

        await vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: 'Building project...', cancellable: false },
            async (progress) => {
                progress.report({ message: 'Starting build...' });

                try {
                    await this.runBuildCommand(sourcePath, workspaceRoot);

                    progress.report({ message: 'Building... waiting for output (this may take 30s-2min)...' });

                    for (let i = 0; i < 120; i++) {
                        await new Promise((r) => setTimeout(r, 1000));

                        builtBinaryPath = await this.findExistingBinary(sourcePath, workspaceRoot);
                        if (builtBinaryPath) break;

                        if (i % 5 === 0) {
                            progress.report({ message: `Still building... (${i}s)` });
                        }
                    }
                } catch (err: any) {
                    vscode.window.showErrorMessage(`Build failed: ${err}`);
                }
            }
        );

        if (!builtBinaryPath) {
            builtBinaryPath = await this.findRecentExeAnywhere(workspaceRoot);
        }

        if (!builtBinaryPath || !fs.existsSync(builtBinaryPath)) {
            vscode.window
                .showErrorMessage('Build completed but no .exe found. Check your build output in dist/ or build/ folders.', 'Open Folder')
                .then((sel) => {
                    if (sel === 'Open Folder') {
                        const distPath = path.join(workspaceRoot, 'dist');
                        if (fs.existsSync(distPath)) {
                            vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(distPath));
                        } else {
                            vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(workspaceRoot));
                        }
                    }
                });
            return null;
        }

        vscode.window.showInformationMessage(`Build complete! Found: ${path.basename(builtBinaryPath)}`);
        return this.scanBinary(builtBinaryPath, showProgress);
    }

    private async runBuildCommand(sourcePath: string, workspaceRoot: string): Promise<void> {
        const ext = path.extname(sourcePath).toLowerCase();

        if (ext === '.js' || ext === '.ts') {
            const pkgPath = path.join(workspaceRoot, 'package.json');
            if (fs.existsSync(pkgPath)) {
                const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

                const hasPkg = pkg.devDependencies?.pkg || pkg.dependencies?.pkg;
                const hasNexe = pkg.devDependencies?.nexe || pkg.dependencies?.nexe;

                if (hasPkg) {
                    return new Promise((resolve, reject) => {
                        const child = spawn('npx', ['pkg', '.', '--output', 'dist/app.exe'], {
                            cwd: workspaceRoot,
                            shell: true,
                            stdio: 'pipe'
                        });

                        let errorOutput = '';
                        child.stderr.on('data', (data: Buffer) => {
                            errorOutput += data;
                        });

                        child.on('close', (code: number | null) => {
                            if (code === 0 || fs.existsSync(path.join(workspaceRoot, 'dist', 'app.exe'))) {
                                resolve();
                            } else {
                                reject(new Error(`pkg failed: ${errorOutput}`));
                            }
                        });
                    });
                } else if (hasNexe) {
                    return new Promise((resolve, reject) => {
                        const child = spawn('npx', ['nexe', sourcePath, '-o', 'dist/app.exe'], {
                            cwd: workspaceRoot,
                            shell: true
                        });
                        child.on('close', (code: number | null) => (code === 0 ? resolve() : reject(new Error('nexe failed'))));
                    });
                } else if (pkg.scripts?.build) {
                    return new Promise((resolve, reject) => {
                        const child = spawn('npm', ['run', 'build'], {
                            cwd: workspaceRoot,
                            shell: true
                        });
                        child.on('close', (code: number | null) => (code === 0 ? resolve() : reject(new Error('npm build failed'))));
                    });
                }
            }

            return new Promise((resolve, reject) => {
                const outDir = path.join(workspaceRoot, 'dist');
                if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

                const child = spawn('npx', ['pkg', sourcePath, '--target', 'node18-win-x64', '--output', 'dist/app.exe'], {
                    cwd: workspaceRoot,
                    shell: true,
                    stdio: 'pipe'
                });

                let errorOutput = '';
                child.stderr.on('data', (data: Buffer) => {
                    errorOutput += data;
                });

                child.on('close', (code: number | null) => {
                    if (code === 0 || fs.existsSync(path.join(workspaceRoot, 'dist', 'app.exe'))) {
                        resolve();
                    } else {
                        reject(new Error(`pkg failed. Try installing it: npm install -g pkg\n${errorOutput}`));
                    }
                });
            });
        } else if (ext === '.py') {
            return new Promise((resolve, reject) => {
                const child = spawn('pyinstaller', ['--onefile', sourcePath, '--distpath', 'dist'], {
                    cwd: workspaceRoot,
                    shell: true
                });
                child.on('close', (code: number | null) => (code === 0 ? resolve() : reject(new Error('pyinstaller failed'))));
            });
        } else if (ext === '.rs') {
            return new Promise((resolve, reject) => {
                const child = spawn('cargo', ['build', '--release'], {
                    cwd: workspaceRoot,
                    shell: true
                });
                child.on('close', (code: number | null) => (code === 0 ? resolve() : reject(new Error('cargo build failed'))));
            });
        } else if (['.cpp', '.c'].includes(ext)) {
            return new Promise((resolve, reject) => {
                const outFile = path.join(workspaceRoot, 'dist', 'app.exe');
                if (!fs.existsSync(path.dirname(outFile))) {
                    fs.mkdirSync(path.dirname(outFile), { recursive: true });
                }

                const child = spawn('g++', [sourcePath, '-o', outFile], {
                    cwd: workspaceRoot,
                    shell: true
                });
                child.on('close', (code: number | null) => (code === 0 ? resolve() : reject(new Error('g++ failed'))));
            });
        }

        throw new Error(`No build command for ${ext}`);
    }

    private async findExistingBinary(sourcePath: string, workspaceRoot: string): Promise<string | null> {
        const ext = path.extname(sourcePath).toLowerCase();
        const baseName = path.basename(sourcePath, ext);

        const checkPaths = [
            path.join(workspaceRoot, 'dist', `${baseName}.exe`),
            path.join(workspaceRoot, 'dist', 'app.exe'),
            path.join(workspaceRoot, 'dist', 'index.exe'),
            path.join(workspaceRoot, 'dist', 'main.exe'),
            path.join(workspaceRoot, 'build', `${baseName}.exe`),
            path.join(workspaceRoot, 'build', 'app.exe'),
            path.join(workspaceRoot, `${baseName}.exe`),
            path.join(workspaceRoot, 'target', 'release', `${baseName}.exe`),
            path.join(workspaceRoot, 'target', 'release', 'app.exe'),
            path.join(workspaceRoot, 'target', 'debug', `${baseName}.exe`)
        ];

        for (const p of checkPaths) {
            if (fs.existsSync(p)) return p;
        }

        if (ext === '.js' || ext === '.ts') {
            const pkgPath = path.join(workspaceRoot, 'package.json');
            if (fs.existsSync(pkgPath)) {
                try {
                    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
                    if (pkg.name) {
                        const nameExe = path.join(workspaceRoot, 'dist', `${pkg.name}.exe`);
                        if (fs.existsSync(nameExe)) return nameExe;
                    }
                    if (pkg.bin) {
                        const binName = Object.keys(pkg.bin)[0];
                        if (binName) {
                            const binExe = path.join(workspaceRoot, 'dist', `${binName}.exe`);
                            if (fs.existsSync(binExe)) return binExe;
                        }
                    }
                } catch (e: any) {
                    // ignore
                }
            }
        }

        return null;
    }

    private async findRecentExeAnywhere(workspaceRoot: string): Promise<string | null> {
        const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;

        const searchDirs = [
            path.join(workspaceRoot, 'dist'),
            path.join(workspaceRoot, 'build'),
            path.join(workspaceRoot, 'target', 'release'),
            path.join(workspaceRoot, 'target', 'debug'),
            path.join(workspaceRoot, 'bin'),
            workspaceRoot
        ];

        let newestExe: string | null = null;
        let newestTime = 0;

        for (const dir of searchDirs) {
            if (!fs.existsSync(dir)) continue;

            try {
                const files = fs.readdirSync(dir);
                for (const file of files) {
                    if (file.endsWith('.exe')) {
                        const fullPath = path.join(dir, file);
                        const stats = fs.statSync(fullPath);
                        if (stats.mtimeMs > fiveMinutesAgo && stats.mtimeMs > newestTime) {
                            newestTime = stats.mtimeMs;
                            newestExe = fullPath;
                        }
                    }
                }
            } catch (e: any) {
                // ignore
            }
        }

        return newestExe;
    }

    private async scanBinary(binaryPath: string, showProgress: boolean = true): Promise<any> {
        const doScan = async (progress?: vscode.Progress<{ message?: string; increment?: number }>) => {
            try {
                progress?.report({ message: 'Sending to DAML API...' });
                
                const result = await this.apiRequest('/predict_path', 'POST', { file_path: binaryPath });

                if (result.is_malicious) {
                    this.threatCount++;
                    this.recordDetection();
                    this.recentDetections.unshift({
                        file: path.basename(binaryPath),
                        severity: result.confidence === 'high' ? 'critical' : result.confidence,
                        time: 'just now',
                        prob: result.malicious_probability
                    });
                    this.recentDetections = this.recentDetections.slice(0, 10);
                    this.updateStatusBar();
                    this.updateWebview();

                    vscode.window.showWarningMessage(
                        `⚠️ Threat in ${path.basename(binaryPath)}! (${(result.malicious_probability * 100).toFixed(0)}% confidence)`
                    );
                } else {
                    if (showProgress) {
                        vscode.window.showInformationMessage(
                            `✅ ${path.basename(binaryPath)} safe (${(result.malicious_probability * 100).toFixed(1)}% risk)`
                        );
                    }
                }
                return result;
            } catch (err: any) {
                if (showProgress) vscode.window.showErrorMessage(`Scan failed: ${err.message || err}`);
                throw err;
            }
        };

        if (showProgress) {
            return vscode.window.withProgress(
                { location: vscode.ProgressLocation.Notification, title: `Scanning ${path.basename(binaryPath)}...` },
                doScan
            );
        } else {
            return doScan();
        }
    }

    private async extractFeatures(filePath: string): Promise<number[]> {
        const pythonDir = path.join(this.context.extensionPath, 'python');
        const scriptPath = path.join(pythonDir, 'extract_features.py');
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

        try {
            const { stdout, stderr } = await execPromise(
                `"${pythonCmd}" "${scriptPath}" "${filePath}"`,
                { timeout: 30000, maxBuffer: 1024 * 1024 * 2 }
            );

            if (stderr && !stderr.includes('DeprecationWarning')) {
                console.warn('Feature extraction stderr:', stderr);
            }

            const features = JSON.parse(stdout.trim());
            if (!Array.isArray(features)) {
                throw new Error(`Invalid output: expected array, got ${typeof features}`);
            }
            if (features.length === 0) {
                throw new Error('Feature extraction returned empty array');
            }

            return features;
        } catch (error: any) {
            if (error.message?.includes('ModuleNotFoundError') || error.stderr?.includes('No module named')) {
                throw new Error(
                    'EMBER Python dependencies missing. Run:\n' +
                        'pip install git+https://github.com/elastic/ember.git\n' +
                        'pip install lief numpy'
                );
            }
            if (error.message?.includes('Not a valid PE') || error.message?.includes('could not extract')) {
                throw new Error(`File is not a valid PE executable: ${path.basename(filePath)}`);
            }
            throw new Error(`Feature extraction failed: ${error.message || error}`);
        }
    }

    private recordDetection() {
        this.rollingHistory[this.rollingHistory.length - 1]++;
        this.context.globalState.update('daml.rollingHistory', this.rollingHistory);
    }

    private shiftRollingWindow() {
        this.rollingHistory.shift();
        this.rollingHistory.push(0);
        this.context.globalState.update('daml.rollingHistory', this.rollingHistory);
        this.updateWebview();
    }

    private resolveThreats() {
        this.threatCount = 0;
        this.recentDetections = [];
        this.rollingHistory = new Array(12).fill(0);
        this.context.globalState.update('daml.rollingHistory', this.rollingHistory);
        this.updateStatusBar();
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: All threats resolved.');
    }

    private resetGraph() {
        this.rollingHistory = new Array(12).fill(0);
        this.context.globalState.update('daml.rollingHistory', this.rollingHistory);
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: Activity graph reset.');
    }

    private updateStatusBar() {
        this.statusBarItem.text = this.isEnabled
            ? `$(shield) DAML ${this.apiAvailable ? '●' : '○'}`
            : `$(circle-slash) DAML Off`;
        this.statusBarItem.tooltip = this.apiAvailable
            ? `API Connected - ${this.threatCount} threats detected`
            : 'API Disconnected - Start with: uvicorn main:app --host 0.0.0.0 --port 8000';
        this.statusBarItem.backgroundColor =
            this.isEnabled && this.threatCount > 0 ? new vscode.ThemeColor('statusBarItem.warningBackground') : undefined;
    }

    private updateWebview() {
        if (!this._view) return;
        if (!this.isEnabled) {
            this._view.webview.html = '<body style="display:flex;justify-content:center;align-items:center;height:100vh;opacity:0.5;">DAML is disabled</body>';
            return;
        }
        const scriptUri = this._view.webview.asWebviewUri(
            vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
        );
        this._view.webview.html = this.getHtml(this._view.webview, scriptUri);
    }

    private getSeverityColor(severity: string): string {
        switch (severity) {
            case 'critical':
                return 'text-red-500';
            case 'high':
                return 'text-orange-500';
            case 'medium':
                return 'text-yellow-500';
            case 'low':
                return 'text-blue-400';
            default:
                return 'text-gray-400';
        }
    }

    private getSeverityIcon(severity: string): string {
        switch (severity) {
            case 'critical':
                return 'shield-alert';
            case 'high':
                return 'alert-triangle';
            case 'medium':
                return 'alert-circle';
            case 'low':
                return 'info';
            default:
                return 'file';
        }
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri) {
        const detectionsList = this.recentDetections
            .map(
                (d) => '\n            <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">\n                <div class="flex items-center gap-2 overflow-hidden">\n                    <i data-lucide="' + this.getSeverityIcon(d.severity) + '" class="w-3 h-3 ' + this.getSeverityColor(d.severity) + ' flex-shrink-0"></i>\n                    <div class="flex flex-col overflow-hidden">\n                        <span class="truncate opacity-90">' + d.file + '</span>\n                        <span class="text-[9px] opacity-50">' + (d.prob * 100).toFixed(1) + '% confidence</span>\n                    </div>\n                </div>\n                <span class="opacity-50 flex-shrink-0 ml-2 text-[9px]">' + d.time + '</span>\n            </li>\n        '
            )
            .join('');

        const emptyState = '\n            <li class="p-4 text-center opacity-50 text-[10px]">\n                <i data-lucide="shield-check" class="w-8 h-8 mx-auto mb-2 text-green-500"></i>\n                No threats detected\n            </li>\n        ';

        const apiStatusClass = this.apiAvailable ? 'text-green-400' : 'text-red-400';
        const apiStatusPulse = this.apiAvailable ? '' : 'pulse';
        const apiStatusText = this.apiAvailable ? '' : '<span class="text-[9px] text-red-400 ml-2">Offline</span>';

        return '<!DOCTYPE html>\n        <html>\n        <head>\n            <script src="https://cdn.tailwindcss.com"></script>\n            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n            <script src="https://unpkg.com/lucide@latest"></script>\n            <style>\n                :root {\n                    --bg: var(--vscode-sideBar-background);\n                    --fg: var(--vscode-foreground);\n                    --border: var(--vscode-widget-border);\n                    --card-bg: var(--vscode-editor-background);\n                    --hover-bg: var(--vscode-list-hoverBackground);\n                }\n                body { background: var(--bg); color: var(--fg); font-family: var(--vscode-font-family); padding: 12px; overflow-x: hidden; }\n                .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; }\n                .chart-container { height: 160px; width: 100%; position: relative; }\n                ::-webkit-scrollbar { width: 6px; }\n                ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }\n                .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }\n                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }\n            </style>\n        </head>\n        <body class="flex flex-col h-[100vh]">\n            <div class="flex flex-col space-y-4 flex-none">\n                <div class="flex justify-between items-center border-b border-[var(--border)] pb-2">\n                    <div class="flex items-center gap-2">\n                        <i data-lucide="shield" class="w-4 h-4 ' + apiStatusClass + ' ' + apiStatusPulse + '"></i>\n                        <h1 class="text-xs font-bold uppercase tracking-wider opacity-80">DAML Status</h1>\n                        ' + apiStatusText + '\n                    </div>\n                    <button id="clear-btn" class="bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded text-[10px] ' + (this.threatCount === 0 ? 'opacity-50' : '') + '" ' + (this.threatCount === 0 ? 'disabled' : '') + '>Clear</button>\n                </div>\n\n                <div class="grid grid-cols-2 gap-2">\n                    <div class="card p-3 flex flex-col items-center justify-center text-center">\n                        <p class="text-[10px] opacity-60 uppercase mb-1"><i data-lucide="alert-triangle" class="w-3 h-3 ' + (this.threatCount > 0 ? 'text-red-500' : 'text-green-500') + ' inline"></i> Threats</p>\n                        <p class="text-2xl font-bold ' + (this.threatCount > 0 ? 'text-red-500' : 'text-green-500') + '">' + this.threatCount + '</p>\n                    </div>\n                    <div class="card p-3 flex flex-col items-center justify-center text-center">\n                        <p class="text-[10px] opacity-60 uppercase mb-1"><i data-lucide="activity" class="w-3 h-3 ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + ' inline"></i> API</p>\n                        <p class="text-2xl font-bold ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + '">' + (this.apiAvailable ? 'ON' : 'OFF') + '</p>\n                    </div>\n                </div>\n\n                <div class="card p-2 chart-container">\n                    <div class="flex justify-between items-center mb-1">\n                        <span class="text-[9px] opacity-60 uppercase">Live Activity (60m)</span>\n                        <button id="reset-graph-btn" class="text-[9px] px-1.5 py-0.5 rounded bg-[var(--border)] hover:opacity-80 transition opacity-60">Reset</button>\n                    </div>\n                    <canvas id="rollingChart"></canvas>\n                </div>\n\n                <div>\n                    <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Actions</h2>\n                    <div class="grid grid-cols-2 gap-2">\n                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanPeFile\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="shield" class="w-4 h-4 text-blue-400"></i>\n                            <span>Scan PE File</span>\n                        </button>\n                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanWorkspace\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="folder-search" class="w-4 h-4 text-blue-400"></i>\n                            <span>Scan Workspace</span>\n                        </button>\n                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanActiveFile\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="file-search" class="w-4 h-4 text-purple-400"></i>\n                            <span>Scan Active File</span>\n                        </button>\n                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer" onclick="vscode.postMessage({command: \'resolveThreats\'})">\n                            <i data-lucide="check-circle" class="w-4 h-4 text-green-400"></i>\n                            <span>Clear Threats</span>\n                        </button>\n                    </div>\n                </div>\n            </div>\n\n            <div class="mt-4 flex flex-col flex-grow overflow-hidden">\n                <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Recent Detections</h2>\n                <div class="card p-0 overflow-y-auto flex-grow h-32">\n                    <ul class="text-[10px] divide-y divide-[var(--border)]">\n                        ' + (this.recentDetections.length > 0 ? detectionsList : emptyState) + '\n                    </ul>\n                </div>\n            </div>\n\n            <div class="mt-3 flex justify-between items-center text-[9px] opacity-50 border-t border-[var(--border)] pt-2 flex-none">\n                <span>EMBER-LSTM Engine</span>\n                <span><i data-lucide="' + (this.apiAvailable ? 'check-circle' : 'x-circle') + '" class="w-3 h-3 ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + ' inline"></i> ' + (this.apiAvailable ? 'Ready' : 'Disconnected') + '</span>\n            </div>\n\n            <script>\n                const vscode = acquireVsCodeApi();\n                document.getElementById(\'clear-btn\')?.addEventListener(\'click\', () => vscode.postMessage({command: \'resolveThreats\'}));\n                document.getElementById(\'reset-graph-btn\')?.addEventListener(\'click\', () => vscode.postMessage({command: \'resetGraph\'}));\n                lucide.createIcons();\n                const slots = Array.from({length: 12}, (_, i) => \'-\' + ((11-i)*5) + \'m\');\n                new Chart(document.getElementById(\'rollingChart\'), {\n                    type: \'line\',\n                    data: { labels: slots, datasets: [{ label: \'Threats\', data: [' + this.rollingHistory.join(',') + '], borderColor: \'rgb(239,68,68)\', backgroundColor: \'rgba(239,68,68,0.1)\', fill: true, tension: 0.4, pointRadius: 3, pointHoverRadius: 5 }] },\n                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: true, ticks: { maxTicksLimit: 6, font: { size: 9 }, color: \'var(--fg)\' }, grid: { display: false } }, y: { display: true, beginAtZero: true, ticks: { font: { size: 9 }, color: \'var(--fg)\', maxTicksLimit: 4, stepSize: 1 }, grid: { color: \'var(--border)\' } } } }\n                });\n            </script>\n            <script src="' + scriptUri.toString() + '"></script>\n        </body>\n        </html>';
    }
}

export function deactivate() {}