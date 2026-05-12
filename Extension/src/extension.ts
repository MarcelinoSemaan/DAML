import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as http from 'http';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

// ── Compiler Configuration Types ───────────────────────────────────────────

type CompilerMode = 'windows-native' | 'linux-cross' | 'custom';

type CompilerConfig = {
    mode: CompilerMode;
    cCompiler: string;
    cppCompiler: string;
    rustTarget: string;
    customPath?: string;
    extraFlags: string[];
};

const DEFAULT_COMPILERS: Record<CompilerMode, CompilerConfig> = {
    'windows-native': {
        mode: 'windows-native',
        cCompiler: 'gcc',
        cppCompiler: 'g++',
        rustTarget: 'x86_64-pc-windows-msvc',
        extraFlags: []
    },
    'linux-cross': {
        mode: 'linux-cross',
        cCompiler: 'x86_64-w64-mingw32-gcc',
        cppCompiler: 'x86_64-w64-mingw32-g++',
        rustTarget: 'x86_64-pc-windows-gnu',
        extraFlags: ['-static-libgcc', '-static-libstdc++']
    },
    'custom': {
        mode: 'custom',
        cCompiler: 'gcc',
        cppCompiler: 'g++',
        rustTarget: 'x86_64-pc-windows-msvc',
        extraFlags: []
    }
};

// ── Activation ─────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
    const provider = new DamlDashboardProvider(context);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('daml.dashboardView', provider),
        vscode.commands.registerCommand('daml.toggleStatusView', () => provider.toggleExtension()),
        vscode.commands.registerCommand('daml.showMenu', () => provider.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.scanWorkspace', () => provider.scanWorkspace()),
        vscode.commands.registerCommand('daml.scanActiveFile', () => provider.scanActiveFile()),
        vscode.commands.registerCommand('daml.scanPeFile', () => provider.scanPeFilePicker()),
        vscode.commands.registerCommand('daml.scanJsonClipboard', () => provider.scanJsonClipboard()),
        vscode.commands.registerCommand('daml.selectCompiler', () => provider.selectCompiler()),
        vscode.commands.registerCommand('daml.forceRebuild', () => provider.forceRebuildActiveFile())
    );
}

// ── Dashboard Provider ─────────────────────────────────────────────────────

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 0;
    private isEnabled: boolean = true;
    private recentDetections: Array<{ file: string; severity: string; time: string; prob: number }> = [];
    private lastExplanation: any = null;
    private rollingHistory: number[] = new Array(12).fill(0);
    private apiAvailable: boolean = false;
    private compilerConfig: CompilerConfig;

    constructor(private context: vscode.ExtensionContext) {
        this.rollingHistory = this.context.globalState.get<number[]>('daml.rollingHistory', new Array(12).fill(0));
        
        const savedConfig = this.context.globalState.get<CompilerConfig>('daml.compilerConfig');
        this.compilerConfig = savedConfig || DEFAULT_COMPILERS['windows-native'];
        
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'daml.showMenu';
        this.updateStatusBar();
        this.statusBarItem.show();

        this.checkApiHealth();
        setInterval(() => this.checkApiHealth(), 5000);
        setInterval(() => this.shiftRollingWindow(), 300000);
    }

    // ── Compiler Selection ─────────────────────────────────────────────────

    public async selectCompiler() {
        const modes: { label: string; description: string; mode: CompilerMode }[] = [
            {
                label: '$(window) Windows Native',
                description: 'Use installed MinGW/GCC on Windows (gcc/g++)',
                mode: 'windows-native'
            },
            {
                label: '$(terminal-linux) Linux Cross-Compile',
                description: 'Use MinGW-w64 cross-compiler (x86_64-w64-mingw32-gcc)',
                mode: 'linux-cross'
            },
            {
                label: '$(gear) Custom',
                description: 'Specify your own compiler paths and flags',
                mode: 'custom'
            }
        ];

        const currentMode = this.compilerConfig.mode;
        const items = modes.map(m => ({
            ...m,
            description: m.mode === currentMode ? `${m.description} (current)` : m.description,
            picked: m.mode === currentMode
        }));

        const selection = await vscode.window.showQuickPick(items, {
            title: 'Select Compiler Mode',
            placeHolder: `Current: ${currentMode}`
        });

        if (!selection) return;

        if (selection.mode === 'custom') {
            await this.configureCustomCompiler();
        } else {
            this.compilerConfig = DEFAULT_COMPILERS[selection.mode];
            await this.context.globalState.update('daml.compilerConfig', this.compilerConfig);
            vscode.window.showInformationMessage(`Compiler set to: ${selection.label.split(' ').slice(1).join(' ')}`);
        }
        
        this.updateStatusBar();
    }

    private async configureCustomCompiler() {
        const cCompiler = await vscode.window.showInputBox({
            prompt: 'C compiler command (e.g., gcc, clang, x86_64-w64-mingw32-gcc)',
            value: this.compilerConfig.cCompiler || 'gcc'
        });
        if (!cCompiler) return;

        const cppCompiler = await vscode.window.showInputBox({
            prompt: 'C++ compiler command (e.g., g++, clang++, x86_64-w64-mingw32-g++)',
            value: this.compilerConfig.cppCompiler || 'g++'
        });
        if (!cppCompiler) return;

        const rustTarget = await vscode.window.showInputBox({
            prompt: 'Rust target triple (e.g., x86_64-pc-windows-msvc)',
            value: this.compilerConfig.rustTarget || 'x86_64-pc-windows-msvc'
        });
        if (!rustTarget) return;

        const extraFlags = await vscode.window.showInputBox({
            prompt: 'Extra compiler flags (space-separated, optional)',
            value: this.compilerConfig.extraFlags.join(' ')
        });

        this.compilerConfig = {
            mode: 'custom',
            cCompiler,
            cppCompiler,
            rustTarget,
            extraFlags: extraFlags ? extraFlags.split(/\s+/) : [],
            customPath: undefined
        };

        await this.context.globalState.update('daml.compilerConfig', this.compilerConfig);
        vscode.window.showInformationMessage(`Custom compiler configured: ${cCompiler} / ${cppCompiler}`);
    }

    private async verifyCompiler(cmd: string): Promise<boolean> {
        try {
            await execPromise(`${cmd} --version`, { timeout: 5000 });
            return true;
        } catch {
            return false;
        }
    }

    private async getAvailableCompiler(sourcePath: string): Promise<{ cmd: string; isCross: boolean } | null> {
        const ext = path.extname(sourcePath).toLowerCase();
        const isCpp = ext === '.cpp';
        
        let preferredCmd = isCpp ? this.compilerConfig.cppCompiler : this.compilerConfig.cCompiler;
        const isCross = this.compilerConfig.mode === 'linux-cross';

        if (await this.verifyCompiler(preferredCmd)) {
            return { cmd: preferredCmd, isCross };
        }

        const fallbacks = isCpp 
            ? ['g++', 'clang++', 'x86_64-w64-mingw32-g++', 'c++']
            : ['gcc', 'clang', 'x86_64-w64-mingw32-gcc', 'cc'];

        for (const cmd of fallbacks) {
            if (await this.verifyCompiler(cmd)) {
                vscode.window.showWarningMessage(
                    `Preferred compiler ${preferredCmd} not found. Using fallback: ${cmd}`
                );
                return { cmd, isCross: cmd.includes('mingw32') };
            }
        }

        return null;
    }

    // ── API Health ─────────────────────────────────────────────────────────

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

    private async apiExplain(filePath: string): Promise<any> {
        return this.apiRequest('/explain', 'POST', { file_path: filePath });
    }

    // ── Webview ──────────────────────────────────────────────────────────

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

    // ── Quick Pick Menu ────────────────────────────────────────────────────

    public async showQuickPickMenu() {
        const toggleLabel = this.isEnabled ? '$(circle-slash) Disable' : '$(play) Enable';
        const compilerLabel = `$(tools) Compiler: ${this.compilerConfig.mode}`;
        
        const items = [
            { label: '$(graph) Open Dashboard', cmd: 'daml.dashboardView.focus' },
            { label: '$(shield) Scan PE File...', cmd: 'daml.scanPeFile' },
            { label: '$(file-json) Scan JSON from Clipboard', cmd: 'daml.scanJsonClipboard' },
            { label: '$(shield) Scan Workspace (Build+Scan)', cmd: 'daml.scanWorkspace' },
            { label: '$(file-code) Scan Active File', cmd: 'daml.scanActiveFile' },
            { label: '$(refresh) Force Rebuild Active File', cmd: 'daml.forceRebuild' },
            { label: compilerLabel, cmd: 'daml.selectCompiler' },
            { label: toggleLabel, cmd: 'daml.toggleStatusView' }
        ];

        const selection = await vscode.window.showQuickPick(items, { title: 'DAML Control' });
        if (selection) vscode.commands.executeCommand(selection.cmd);
    }

    // ── Scan Commands ──────────────────────────────────────────────────────

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
            // ── FIX 1: Also look for existing EXEs ──
            const exeFiles = await vscode.workspace.findFiles('**/*.exe', '**/node_modules/**', 50);
            const sourceFiles = await vscode.workspace.findFiles('**/*.{ts,js,py,cpp,c}', '**/node_modules/**', 50);

            if (exeFiles.length === 0 && sourceFiles.length === 0) {
                vscode.window.showInformationMessage('No source files or compiled EXEs found.');
                return;
            }

            let scanned = 0;
            const threats: typeof this.recentDetections = [];
            const totalFiles = exeFiles.length + sourceFiles.length;

            // Scan existing EXEs directly (no rebuild needed)
            for (const file of exeFiles) {
                if (token.isCancellationRequested) break;
                progress.report({ message: `Scanning ${path.basename(file.fsPath)}...`, increment: 100 / totalFiles });

                try {
                    const result = await this.scanBinary(file.fsPath, false);
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

            // Build + scan source files
            for (const file of sourceFiles) {
                if (token.isCancellationRequested) break;
                progress.report({ message: `Building ${path.basename(file.fsPath)}...`, increment: 100 / totalFiles });

                try {
                    // ── FIX 2: buildAndScan now actually works in batch mode ──
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



    public async forceRebuildActiveFile() {
        await this.scanActiveFile();
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

                        vscode.window.showWarningMessage(
                            `⚠️ Threat detected in clipboard JSON! (${(result.malicious_probability * 100).toFixed(0)}% confidence)`
                        );
                    } else {
                        vscode.window.showInformationMessage(
                            `✅ Clipboard JSON safe (${(result.malicious_probability * 100).toFixed(1)}% risk)`
                        );
                    }
                    this.updateWebview();
                    return result;
                } catch (err: any) {
                    vscode.window.showErrorMessage(`Scan failed: ${err.message || err}`);
                    throw err;
                }
            }
        );
    }

    // ── Build & Scan ───────────────────────────────────────────────────────

    private async buildAndScan(sourcePath: string, showProgress: boolean): Promise<any | null> {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath || '';
    const fileName = path.basename(sourcePath);
    const ext = path.extname(sourcePath).toLowerCase();
    const baseName = path.basename(sourcePath, ext);
    const outFile = path.join(workspaceRoot, 'dist', `${baseName}.exe`);

    // ── FIX: Only skip the interactive dialog in batch mode, not the whole build ──
    if (showProgress) {
        const action = await vscode.window.showWarningMessage(
            `Build ${fileName} into EXE and scan?`, 
            'Build & Scan', 
            'Cancel'
        );
        if (action !== 'Build & Scan') return null;
    }

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: `Building ${fileName}...`, cancellable: false },
        async (progress) => {
            progress.report({ message: 'Compiling...' });
            try {
                await this.runBuildCommand(sourcePath, workspaceRoot);
            } catch (err: any) {
                vscode.window.showErrorMessage(`Build failed: ${err.message || err}`);
                throw err;
            }
        }
    );

    if (!fs.existsSync(outFile)) {
        vscode.window.showErrorMessage(`No .exe produced. Expected: dist/${baseName}.exe`);
        return null;
    }

    return this.scanBinary(outFile, showProgress);
}

    // ── Build Commands (with compiler selection) ─────────────────────────

    private async runBuildCommand(sourcePath: string, workspaceRoot: string): Promise<void> {
        const ext = path.extname(sourcePath).toLowerCase();

        if (ext === '.js' || ext === '.ts') {
            return this.runJsBuild(sourcePath, workspaceRoot);
        } else if (ext === '.py') {
            return this.runPythonBuild(sourcePath, workspaceRoot);
        } else if (ext === '.rs') {
            return this.runRustBuild(sourcePath, workspaceRoot);
        } else if (['.cpp', '.c'].includes(ext)) {
            return this.runCppBuild(sourcePath, workspaceRoot);
        }

        throw new Error(`No build command for ${ext}`);
    }

    private async runCppBuild(sourcePath: string, workspaceRoot: string): Promise<void> {
        const compiler = await this.getAvailableCompiler(sourcePath);
        if (!compiler) {
            throw new Error('No compiler available');
        }

        const ext = path.extname(sourcePath).toLowerCase();
        const baseName = path.basename(sourcePath, ext);
        const outFile = path.join(workspaceRoot, 'dist', `${baseName}.exe`);

        if (!fs.existsSync(path.dirname(outFile))) {
            fs.mkdirSync(path.dirname(outFile), { recursive: true });
        }

        const flags = ['-O2', '-Wall', ...this.compilerConfig.extraFlags];

        if (compiler.isCross || this.compilerConfig.mode === 'linux-cross') {
            flags.push('-static-libgcc');
            if (ext === '.cpp') {
                flags.push('-static-libstdc++');
            }
        }

        return new Promise((resolve, reject) => {
            const args = [sourcePath, ...flags, '-o', outFile];
            const child = spawn(compiler.cmd, args, {
                cwd: workspaceRoot,
                shell: true,
                stdio: 'pipe'
            });

            let errorOutput = '';
            child.stderr.on('data', (data: Buffer) => {
                errorOutput += data.toString();
            });

            child.on('close', (code: number | null) => {
                if (code === 0 && fs.existsSync(outFile)) {
                    resolve();
                } else {
                    reject(new Error(`${compiler.cmd} failed (code ${code}):\n${errorOutput}`));
                }
            });
        });
    }

    private async runRustBuild(sourcePath: string, workspaceRoot: string): Promise<void> {
        const target = this.compilerConfig.rustTarget;
        
        try {
            const { stdout } = await execPromise('rustup target list --installed');
            if (!stdout.includes(target)) {
                const install = await vscode.window.showWarningMessage(
                    `Rust target ${target} not installed. Install now?`,
                    'Yes',
                    'No'
                );
                if (install === 'Yes') {
                    await vscode.window.withProgress(
                        { location: vscode.ProgressLocation.Notification, title: `Installing Rust target ${target}...` },
                        async () => {
                            await execPromise(`rustup target add ${target}`);
                        }
                    );
                }
            }
        } catch {
            // rustup not available, try anyway
        }

        return new Promise((resolve, reject) => {
            const args = ['build', '--release', '--target', target];
            const child = spawn('cargo', args, {
                cwd: workspaceRoot,
                shell: true,
                stdio: 'pipe'
            });

            let errorOutput = '';
            child.stderr.on('data', (data: Buffer) => {
                errorOutput += data.toString();
            });

            child.on('close', (code: number | null) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`cargo build failed (target: ${target}):\n${errorOutput}`));
                }
            });
        });
    }

    private async runJsBuild(sourcePath: string, workspaceRoot: string): Promise<void> {
        const pkgPath = path.join(workspaceRoot, 'package.json');
        const ext = path.extname(sourcePath).toLowerCase();
        const baseName = path.basename(sourcePath, ext);
        const outFile = path.join(workspaceRoot, 'dist', `${baseName}.exe`);

        if (!fs.existsSync(path.dirname(outFile))) {
            fs.mkdirSync(path.dirname(outFile), { recursive: true });
        }

        if (fs.existsSync(pkgPath)) {
            const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

            const hasPkg = pkg.devDependencies?.pkg || pkg.dependencies?.pkg;
            const hasNexe = pkg.devDependencies?.nexe || pkg.dependencies?.nexe;

            if (hasPkg) {
                return new Promise((resolve, reject) => {
                    const child = spawn('npx', ['pkg', '.', '--output', outFile], {
                        cwd: workspaceRoot,
                        shell: true,
                        stdio: 'pipe'
                    });

                    let errorOutput = '';
                    child.stderr.on('data', (data: Buffer) => {
                        errorOutput += data;
                    });

                    child.on('close', (code: number | null) => {
                        if (code === 0 && fs.existsSync(outFile)) {
                            resolve();
                        } else {
                            reject(new Error(`pkg failed: ${errorOutput}`));
                        }
                    });
                });
            } else if (hasNexe) {
                return new Promise((resolve, reject) => {
                    const child = spawn('npx', ['nexe', sourcePath, '-o', outFile], {
                        cwd: workspaceRoot,
                        shell: true
                    });
                    child.on('close', (code: number | null) => {
                        if (code === 0 && fs.existsSync(outFile)) {
                            resolve();
                        } else {
                            reject(new Error('nexe failed'));
                        }
                    });
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
            const child = spawn('npx', ['pkg', sourcePath, '--target', 'node18-win-x64', '--output', outFile], {
                cwd: workspaceRoot,
                shell: true,
                stdio: 'pipe'
            });

            let errorOutput = '';
            child.stderr.on('data', (data: Buffer) => {
                errorOutput += data;
            });

            child.on('close', (code: number | null) => {
                if (code === 0 && fs.existsSync(outFile)) {
                    resolve();
                } else {
                    reject(new Error(`pkg failed. Try installing it: npm install -g pkg\n${errorOutput}`));
                }
            });
        });
    }

    private async runPythonBuild(sourcePath: string, workspaceRoot: string): Promise<void> {
        const ext = path.extname(sourcePath).toLowerCase();
        const baseName = path.basename(sourcePath, ext);
        const outFile = path.join(workspaceRoot, 'dist', `${baseName}.exe`);

        if (!fs.existsSync(path.dirname(outFile))) {
            fs.mkdirSync(path.dirname(outFile), { recursive: true });
        }

        return new Promise((resolve, reject) => {
            const child = spawn('pyinstaller', ['--onefile', sourcePath, '--distpath', 'dist', '-n', baseName], {
                cwd: workspaceRoot,
                shell: true
            });
            child.on('close', (code: number | null) => {
                if (code === 0 && fs.existsSync(outFile)) {
                    resolve();
                } else {
                    reject(new Error('pyinstaller failed'));
                }
            });
        });
    }

    // ── Binary Discovery ───────────────────────────────────────────────────

    private async findBinary(sourcePath: string, workspaceRoot: string): Promise<string | null> {
        const ext = path.extname(sourcePath).toLowerCase();
        const baseName = path.basename(sourcePath, ext);
        const outFile = path.join(workspaceRoot, 'dist', `${baseName}.exe`);
        return fs.existsSync(outFile) ? outFile : null;
    }

    // ── Scanning ───────────────────────────────────────────────────────────

    private async scanBinary(binaryPath: string, showProgress: boolean = true): Promise<any> {
        const doScan = async (progress?: vscode.Progress<{ message?: string; increment?: number }>) => {
            try {
                progress?.report({ message: 'Sending to DAML API...' });
                
                const result = await this.apiRequest('/predict_path', 'POST', { file_path: binaryPath });

                // Fetch explanation
                try {
                    this.lastExplanation = await this.apiRequest('/explain', 'POST', { file_path: binaryPath });
                } catch (e: any) {
                    this.lastExplanation = { error: true, message: e.message || 'Explain failed' };
                }


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
                this.updateWebview();
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

    // ── State Management ─────────────────────────────────────────────────────

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

    // ── UI Updates ───────────────────────────────────────────────────────────

    private updateStatusBar() {
        const compilerIndicator = this.compilerConfig.mode === 'linux-cross' ? '[WSL]' : 
                                  this.compilerConfig.mode === 'custom' ? '[Custom]' : '';
        
        this.statusBarItem.text = this.isEnabled
            ? `$(shield) DAML ${compilerIndicator} ${this.apiAvailable ? '●' : '○'}`
            : `$(circle-slash) DAML Off`;
            
        this.statusBarItem.tooltip = this.apiAvailable
            ? `API Connected | Compiler: ${this.compilerConfig.mode} | ${this.threatCount} threats`
            : `API Disconnected | Compiler: ${this.compilerConfig.mode}\nStart server: uvicorn main:app --host 0.0.0.0 --port 8000`;
            
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

    // ── HTML Generation (Resolution Adaptive) ────────────────────────────────

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
        // Build explanation panel HTML if available
        let explanationHtml = '';
        if (this.lastExplanation) {
            const exp = this.lastExplanation;

            // Error state
            if (exp.error) {
                explanationHtml = `
                <div class="card p-3 mt-3 bg-yellow-500/10 border-l-4 border-l-yellow-500">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="alert-circle" class="w-4 h-4 text-yellow-500"></i>
                        <h3 class="text-responsive font-bold uppercase tracking-wider opacity-80">Explanation Unavailable</h3>
                    </div>
                    <p class="text-responsive-sm opacity-70">${exp.message}</p>
                    <p class="text-responsive-xs opacity-50 mt-2">Make sure your FastAPI server is running the updated main_explain.py with the /explain endpoint.</p>
                </div>
                `;
            } else {
                const probPct = (exp.probability * 100).toFixed(1);
                const predColor = exp.prediction === 'MALICIOUS' ? 'text-red-500' : 'text-green-500';
                const predBg = exp.prediction === 'MALICIOUS' ? 'bg-red-500/10' : 'bg-green-500/10';

                // Top features bars
                const featureBars = exp.top_features.map((f: any) => {
                    const maxAttr = exp.top_features[0].attribution;
                    const pct = maxAttr > 0 ? (f.attribution / maxAttr * 100).toFixed(0) : '0';
                    const dirColor = f.direction === 'malicious' ? 'bg-red-500' : 'bg-green-500';
                    const dirIcon = f.direction === 'malicious' ? '↑' : '↓';
                    return `
                    <div class="mb-2">
                        <div class="flex justify-between text-responsive-xs mb-0.5">
                            <span class="truncate opacity-80" title="${f.feature}">${f.rank}. ${f.feature}</span>
                            <span class="opacity-60">${dirIcon} ${f.group}</span>
                        </div>
                        <div class="w-full bg-[var(--border)] rounded-full h-1.5">
                            <div class="${dirColor} h-1.5 rounded-full transition-all" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
                }).join('');

                // Group contributions
                const groupBars = exp.groups.slice(0, 6).map((g: any) => {
                    const dirColor = g.direction === 'malicious' ? 'text-red-400' : 'text-green-400';
                    return `
                    <div class="flex justify-between items-center text-responsive-xs py-1 border-b border-[var(--border)] last:border-0">
                        <span class="capitalize opacity-80">${g.group.replace(/_/g, ' ')}</span>
                        <span class="${dirColor} font-mono">${g.percentage}%</span>
                    </div>
                `;
                }).join('');

                // ── FIX: Render actionable recommendations instead of raw benign pushers ──
                const recommendationsHtml = exp.recommendations && exp.recommendations.length > 0
                    ? exp.recommendations.map((rec: any) => `
                    <div class="mb-2 p-2 bg-[var(--border)]/30 rounded">
                        <div class="text-responsive-xs font-bold opacity-80 mb-1">${rec.feature} <span class="opacity-50">(${rec.group})</span></div>
                        <div class="text-responsive-xs opacity-70 leading-relaxed">${rec.advice}</div>
                    </div>
                `).join('')
                    : (exp.benign_pushers && exp.benign_pushers.length > 0
                        ? exp.benign_pushers.slice(0, 8).map((bp: any) => `
                        <div class="flex justify-between items-center text-responsive-xs py-1 border-b border-[var(--border)] last:border-0">
                            <span class="opacity-80">${bp.feature}</span>
                            <span class="text-green-400 font-mono">${bp.attribution.toFixed(4)} ↓</span>
                        </div>
                    `).join('')
                        : '<p class="text-responsive-xs opacity-50">No remediation advice available.</p>');

                explanationHtml = `
            <div class="card p-3 mt-3 ${predBg} border-l-4 ${exp.prediction === 'MALICIOUS' ? 'border-l-red-500' : 'border-l-green-500'}">
                <div class="flex justify-between items-center mb-2">
                    <h3 class="text-responsive font-bold uppercase tracking-wider opacity-80">Explanation</h3>
                    <span class="text-responsive-lg font-bold ${predColor}">${exp.prediction}</span>
                </div>
                <div class="mb-3">
                    <div class="flex justify-between text-responsive-sm mb-1">
                        <span class="opacity-60">Confidence</span>
                        <span class="font-mono">${probPct}%</span>
                    </div>
                    <div class="w-full bg-[var(--border)] rounded-full h-2">
                        <div class="${exp.prediction === 'MALICIOUS' ? 'bg-red-500' : 'bg-green-500'} h-2 rounded-full transition-all" style="width: ${probPct}%"></div>
                    </div>
                </div>
                
                <div class="mb-3">
                    <h4 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">Top Influential Features</h4>
                    ${featureBars}
                </div>
                
                <div>
                    <h4 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">Feature Group Impact</h4>
                    ${groupBars}
                </div>
                
                <div class="mt-3">
                    <h4 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">How to Lower the Score</h4>
                    ${recommendationsHtml}
                </div>
            </div>
            `;
            }
        }

        const detectionsList = this.recentDetections
            .map(
                (d) => '\n            <li class="detection-item flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">\n                <div class="flex items-center gap-2 overflow-hidden min-w-0">\n                    <i data-lucide="' + this.getSeverityIcon(d.severity) + '" class="w-3 h-3 ' + this.getSeverityColor(d.severity) + ' flex-shrink-0"></i>\n                    <div class="flex flex-col overflow-hidden min-w-0">\n                        <span class="truncate opacity-90 text-responsive">' + d.file + '</span>\n                        <span class="text-responsive-sm opacity-50">' + (d.prob * 100).toFixed(1) + '% confidence</span>\n                    </div>\n                </div>\n                <span class="opacity-50 flex-shrink-0 ml-2 text-responsive-sm">' + d.time + '</span>\n            </li>\n        '
            )
            .join('');

        const emptyState = '\n            <li class="detection-empty text-center opacity-50">\n                <i data-lucide="shield-check" class="w-8 h-8 mx-auto mb-2 text-green-500 icon-responsive"></i>\n                <span class="text-responsive">No threats detected</span>\n            </li>\n        ';

        const apiStatusClass = this.apiAvailable ? 'text-green-400' : 'text-red-400';
        const apiStatusPulse = this.apiAvailable ? '' : 'pulse';
        const apiStatusText = this.apiAvailable ? '' : '<span class="text-responsive-sm text-red-400 ml-2">Offline</span>';

        return '<!DOCTYPE html>\n        <html>\n        <head>\n            <meta name="viewport" content="width=device-width, initial-scale=1.0">\n            <script src="https://cdn.tailwindcss.com"></script>\n            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n            <script src="https://unpkg.com/lucide@latest"></script>\n            <style>\n                :root {\n                    --bg: var(--vscode-sideBar-background);\n                    --fg: var(--vscode-foreground);\n                    --border: var(--vscode-widget-border);\n                    --card-bg: var(--vscode-editor-background);\n                    --hover-bg: var(--vscode-list-hoverBackground);\n                    --sidebar-width: 100%;\n                }\n                \n                /* Base responsive font scaling */\n                html { font-size: clamp(10px, 0.8vw + 6px, 14px); }\n                \n                body { \n                    background: var(--bg); \n                    color: var(--fg); \n                    font-family: var(--vscode-font-family); \n                    padding: clamp(8px, 1vw, 16px); \n                    overflow-x: hidden; \n                    min-height: 100vh;\n                }\n                \n                .card { \n                    background: var(--card-bg); \n                    border: 1px solid var(--border); \n                    border-radius: clamp(4px, 0.5vw, 8px); \n                }\n                \n                .chart-container { \n                    height: clamp(120px, 25vh, 200px); \n                    width: 100%; \n                    position: relative; \n                }\n                \n                /* Responsive text sizes */\n                .text-responsive { font-size: clamp(0.7rem, 0.8vw + 0.3rem, 1rem); }\n                .text-responsive-sm { font-size: clamp(0.6rem, 0.6vw + 0.2rem, 0.85rem); }\n                .text-responsive-lg { font-size: clamp(1.2rem, 2vw + 0.5rem, 2.5rem); }\n                .text-responsive-xs { font-size: clamp(0.55rem, 0.5vw + 0.15rem, 0.75rem); }\n                \n                /* Responsive spacing */\n                .p-responsive { padding: clamp(6px, 1vw, 12px); }\n                .gap-responsive { gap: clamp(4px, 0.8vw, 8px); }\n                \n                /* Responsive icons */\n                .icon-responsive { \n                    width: clamp(16px, 2vw, 24px); \n                    height: clamp(16px, 2vw, 24px); \n                }\n                .icon-responsive-sm { \n                    width: clamp(12px, 1.5vw, 16px); \n                    height: clamp(12px, 1.5vw, 16px); \n                }\n                \n                /* Detection list items */\n                .detection-item { \n                    padding: clamp(6px, 1vw, 10px); \n                    border-bottom: 1px solid var(--border);\n                }\n                .detection-empty { \n                    padding: clamp(12px, 2vw, 20px); \n                }\n                \n                /* Scrollbar */\n                ::-webkit-scrollbar { width: clamp(4px, 0.6vw, 8px); }\n                ::-webkit-scrollbar-thumb { \n                    background: var(--border); \n                    border-radius: clamp(2px, 0.3vw, 4px); \n                }\n                \n                /* Pulse animation */\n                .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }\n                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }\n                \n                /* Grid responsive */\n                .grid-responsive { \n                    display: grid; \n                    grid-template-columns: repeat(auto-fit, minmax(min(100%, 140px), 1fr)); \n                    gap: clamp(4px, 1vw, 8px); \n                }\n                \n                /* Button responsive */\n                .btn-responsive {\n                    padding: clamp(4px, 0.8vw, 8px) clamp(6px, 1vw, 12px);\n                    font-size: clamp(0.65rem, 0.7vw + 0.2rem, 0.9rem);\n                    border-radius: clamp(3px, 0.4vw, 6px);\n                }\n                \n                /* Status header responsive */\n                .header-responsive {\n                    padding-bottom: clamp(6px, 1vw, 12px);\n                    margin-bottom: clamp(8px, 1.5vw, 16px);\n                }\n                \n                /* Chart canvas responsive */\n                canvas { max-height: 100% !important; }\n                \n                /* Hide on very small */\n                @media (max-width: 200px) {\n                    .hide-narrow { display: none; }\n                }\n                \n                /* Compact mode for very small sidebar */\n                @media (max-width: 180px) {\n                    .compact-hide { display: none; }\n                    .grid-responsive { grid-template-columns: 1fr; }\n                }\n            </style>\n        </head>\n        <body class="flex flex-col min-h-[100vh]">\n            <div class="flex flex-col space-y-4 flex-none">\n                <div class="flex justify-between items-center border-b border-[var(--border)] header-responsive">\n                    <div class="flex items-center gap-2 min-w-0">\n                        <i data-lucide="shield" class="icon-responsive ' + apiStatusClass + ' ' + apiStatusPulse + ' flex-shrink-0"></i>\n                        <h1 class="text-responsive font-bold uppercase tracking-wider opacity-80 truncate hide-narrow">DAML Status</h1>\n                        ' + apiStatusText + '\n                    </div>\n                    <button id="clear-btn" class="bg-blue-600 hover:bg-blue-500 text-white btn-responsive flex-shrink-0 ' + (this.threatCount === 0 ? 'opacity-50' : '') + '" ' + (this.threatCount === 0 ? 'disabled' : '') + '>Clear</button>\n                </div>\n\n                <div class="grid-responsive">\n                    <div class="card p-responsive flex flex-col items-center justify-center text-center">\n                        <p class="text-responsive-xs opacity-60 uppercase mb-1"><i data-lucide="alert-triangle" class="icon-responsive-sm ' + (this.threatCount > 0 ? 'text-red-500' : 'text-green-500') + ' inline"></i> <span class="hide-narrow">Threats</span></p>\n                        <p class="text-responsive-lg font-bold ' + (this.threatCount > 0 ? 'text-red-500' : 'text-green-500') + '">' + this.threatCount + '</p>\n                    </div>\n                    <div class="card p-responsive flex flex-col items-center justify-center text-center">\n                        <p class="text-responsive-xs opacity-60 uppercase mb-1"><i data-lucide="activity" class="icon-responsive-sm ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + ' inline"></i> <span class="hide-narrow">API</span></p>\n                        <p class="text-responsive-lg font-bold ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + '">' + (this.apiAvailable ? 'ON' : 'OFF') + '</p>\n                    </div>\n                </div>\n\n                <div class="card p-2 chart-container">\n                    <div class="flex justify-between items-center mb-1">\n                        <span class="text-responsive-xs opacity-60 uppercase">Live Activity (60m)</span>\n                        <button id="reset-graph-btn" class="text-responsive-xs px-1.5 py-0.5 rounded bg-[var(--border)] hover:opacity-80 transition opacity-60">Reset</button>\n                    </div>\n                    <canvas id="rollingChart"></canvas>\n                </div>\n\n                <div>\n                    <h2 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">Actions</h2>\n                    <div class="grid-responsive">\n                        <button class="card p-responsive text-responsive flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanPeFile\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="shield" class="icon-responsive text-blue-400"></i>\n                            <span class="compact-hide">Scan PE</span>\n                        </button>\n                        <button class="card p-responsive text-responsive flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanWorkspace\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="folder-search" class="icon-responsive text-blue-400"></i>\n                            <span class="compact-hide">Workspace</span>\n                        </button>\n                        <button class="card p-responsive text-responsive flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer ' + (!this.apiAvailable ? 'opacity-50' : '') + '" onclick="vscode.postMessage({command: \'scanActiveFile\'})" ' + (!this.apiAvailable ? 'disabled' : '') + '>\n                            <i data-lucide="file-search" class="icon-responsive text-purple-400"></i>\n                            <span class="compact-hide">Active File</span>\n                        </button>\n                        <button class="card p-responsive text-responsive flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer" onclick="vscode.postMessage({command: \'resolveThreats\'})">\n                            <i data-lucide="check-circle" class="icon-responsive text-green-400"></i>\n                            <span class="compact-hide">Clear</span>\n                        </button>\n                    </div>\n                </div>\n            </div>\n\n            <div class="mt-4 flex flex-col flex-grow overflow-hidden">\n                <h2 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">Recent Detections</h2>\n                <div class="card p-0 overflow-y-auto flex-grow" style="min-height: clamp(80px, 15vh, 150px);">\n                    <ul class="text-responsive divide-y divide-[var(--border)]">\n                        ' + (this.recentDetections.length > 0 ? detectionsList : emptyState) + '\n                    </ul>\n                </div>\n            </div>\n            </div>\n\n            <div class="mt-4 flex flex-col flex-none">\n                <h2 class="text-responsive-xs font-bold uppercase opacity-60 mb-2">Last Scan Explanation</h2>\n                ' + explanationHtml + '\n            </div>\n\n            <div class="mt-3 flex justify-between items-center text-responsive-xs opacity-50 border-t border-[var(--border)] pt-2 flex-none">\n                <span class="hide-narrow">EMBER-LSTM</span>\n                <span><i data-lucide="' + (this.apiAvailable ? 'check-circle' : 'x-circle') + '" class="icon-responsive-sm ' + (this.apiAvailable ? 'text-green-500' : 'text-red-500') + ' inline"></i> ' + (this.apiAvailable ? 'Ready' : 'Disconnected') + '</span>\n            </div>\n\n            <script>\n                const vscode = acquireVsCodeApi();\n                document.getElementById(\'clear-btn\')?.addEventListener(\'click\', () => vscode.postMessage({command: \'resolveThreats\'}));\n                document.getElementById(\'reset-graph-btn\')?.addEventListener(\'click\', () => vscode.postMessage({command: \'resetGraph\'}));\n                lucide.createIcons();\n                \n                // Responsive chart sizing\n                const chartContainer = document.querySelector(\'.chart-container\');\n                const slots = Array.from({length: 12}, (_, i) => \'-\' + ((11-i)*5) + \'m\');\n                \n                const chart = new Chart(document.getElementById(\'rollingChart\'), {\n                    type: \'line\',\n                    data: { \n                        labels: slots, \n                        datasets: [{ \n                            label: \'Threats\', \n                            data: [' + this.rollingHistory.join(',') + '], \n                            borderColor: \'rgb(239,68,68)\', \n                            backgroundColor: \'rgba(239,68,68,0.1)\', \n                            fill: true, \n                            tension: 0.4, \n                            pointRadius: Math.max(2, Math.min(4, window.innerWidth / 200)), \n                            pointHoverRadius: Math.max(3, Math.min(6, window.innerWidth / 150)) \n                        }] \n                    },\n                    options: { \n                        responsive: true, \n                        maintainAspectRatio: false, \n                        plugins: { \n                            legend: { display: false } \n                        }, \n                        scales: { \n                            x: { \n                                display: true, \n                                ticks: { \n                                    maxTicksLimit: Math.max(4, Math.min(8, Math.floor(window.innerWidth / 80))), \n                                    font: { size: Math.max(8, Math.min(11, window.innerWidth / 80)) }, \n                                    color: \'var(--fg)\' \n                                }, \n                                grid: { display: false } \n                            }, \n                            y: { \n                                display: true, \n                                beginAtZero: true, \n                                ticks: { \n                                    font: { size: Math.max(8, Math.min(11, window.innerWidth / 80)) }, \n                                    color: \'var(--fg)\', \n                                    maxTicksLimit: Math.max(3, Math.min(5, Math.floor(window.innerHeight / 100))), \n                                    stepSize: 1 \n                                }, \n                                grid: { color: \'var(--border)\' } \n                            } \n                        } \n                    }\n                });\n                \n                // Resize observer for dynamic chart updates\n                const resizeObserver = new ResizeObserver(entries => {\n                    for (let entry of entries) {\n                        const width = entry.contentRect.width;\n                        chart.options.scales.x.ticks.font.size = Math.max(8, Math.min(11, width / 80));\n                        chart.options.scales.x.ticks.maxTicksLimit = Math.max(4, Math.min(8, Math.floor(width / 80)));\n                        chart.options.scales.y.ticks.font.size = Math.max(8, Math.min(11, width / 80));\n                        chart.update(\'none\');\n                    }\n                });\n                resizeObserver.observe(chartContainer);\n            </script>\n            <script src="' + scriptUri.toString() + '"></script>\n        </body>\n        </html>';
    }
}

export function deactivate() {}