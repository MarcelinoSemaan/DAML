import * as vscode from 'vscode';
import * as path from 'path';
import * as http from 'http';
import { exec } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

// FastAPI configuration
const API_BASE_URL = 'http://localhost:8000';

export function activate(context: vscode.ExtensionContext) {
    const provider = new DamlDashboardProvider(context);

    // Register the sidebar provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('daml.dashboardView', provider)
    );

    // Command to toggle protection from the status bar or menu
    context.subscriptions.push(
        vscode.commands.registerCommand('daml.toggleStatusView', () => provider.toggleExtension()),
        vscode.commands.registerCommand('daml.showMenu', () => provider.showQuickPickMenu()),
        vscode.commands.registerCommand('daml.scanWorkspace', () => provider.scanWorkspace()),
        vscode.commands.registerCommand('daml.scanActiveFile', () => provider.scanActiveFile())
    );
}

class DamlDashboardProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private statusBarItem: vscode.StatusBarItem;
    private threatCount: number = 0;
    private isEnabled: boolean = true;
    private recentDetections: Array<{file: string, severity: string, time: string, prob: number}> = [];
    private apiAvailable: boolean = false;

    constructor(private context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'daml.showMenu';
        this.updateStatusBar();
        this.statusBarItem.show();
        
        // Check API health on startup
        this.checkApiHealth();
    }

    private async checkApiHealth(): Promise<void> {
        try {
            const health = await this.apiRequest('/health');
            this.apiAvailable = health.status === 'ok';
            if (this.apiAvailable) {
                console.log(`✅ DAML API connected: ${health.device}`);
            }
        } catch (e) {
            this.apiAvailable = false;
            console.warn('⚠️ DAML API not available. Make sure FastAPI server is running on port 8000');
        }
    }

    private async apiRequest(endpoint: string, method: 'GET' | 'POST' = 'GET', body?: any): Promise<any> {
        return new Promise((resolve, reject) => {
            const postData = body ? JSON.stringify(body) : '';
            const options = {
                hostname: 'localhost',
                port: 8000,
                path: endpoint,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const req = http.request(options, (res) => {
                let data = '';
                res.on('data', (chunk) => data += chunk);
                res.on('end', () => {
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        reject(new Error('Invalid JSON response'));
                    }
                });
            });

            req.on('error', (err) => reject(err));
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

        webviewView.webview.onDidReceiveMessage(async msg => {
            if (msg.command === 'resolveThreats') this.resolveThreats();
            if (msg.command === 'scanWorkspace') this.scanWorkspace();
            if (msg.command === 'scanActiveFile') this.scanActiveFile();
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
            { label: '$(shield) Scan Workspace', cmd: 'daml.scanWorkspace' },
            { label: '$(file-code) Scan Active File', cmd: 'daml.scanActiveFile' },
            { label: toggleLabel, cmd: 'daml.toggleStatusView' }
        ];

        const selection = await vscode.window.showQuickPick(items, { title: 'DAML Control' });
        if (selection) vscode.commands.executeCommand(selection.cmd);
    }

    public async scanWorkspace() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');
        if (!this.apiAvailable) {
            const action = await vscode.window.showWarningMessage(
                'DAML API server not running. Start it with: uvicorn main:app --host 0.0.0.0 --port 8000',
                'Retry', 'Cancel'
            );
            if (action === 'Retry') {
                await this.checkApiHealth();
                if (this.apiAvailable) this.scanWorkspace();
            }
            return;
        }

        vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: "DAML Scanning Workspace..." }, 
            async (progress) => {
                const files = await vscode.workspace.findFiles('**/*.{exe,dll,elf}', '**/node_modules/**', 100);
                let scanned = 0;
                const threats: typeof this.recentDetections = [];

                for (const file of files) {
                    progress.report({ message: `Scanning ${path.basename(file.fsPath)}...`, increment: 100 / files.length });
                    
                    try {
                        // Extract features using Python subprocess with thrember
                        const features = await this.extractFeatures(file.fsPath);
                        const result = await this.apiRequest('/predict', 'POST', { features });
                        
                        if (result.is_malicious) {
                            threats.push({
                                file: path.basename(file.fsPath),
                                severity: result.confidence === 'high' ? 'critical' : result.confidence,
                                time: 'just now',
                                prob: result.malicious_probability
                            });
                        }
                    } catch (err) {
                        console.error(`Failed to scan ${file.fsPath}:`, err);
                    }
                    
                    scanned++;
                    await new Promise(r => setTimeout(r, 100)); // Rate limiting
                }

                this.threatCount += threats.length;
                this.recentDetections = [...threats, ...this.recentDetections].slice(0, 10);
                
                this.updateStatusBar();
                this.updateWebview();
                
                if (threats.length > 0) {
                    vscode.window.showWarningMessage(`DAML: Found ${threats.length} potential threats!`);
                } else {
                    vscode.window.showInformationMessage(`DAML: Workspace scan complete. No threats detected.`);
                }
            }
        );
    }

    public async scanActiveFile() {
        if (!this.isEnabled) return vscode.window.showWarningMessage('Enable DAML first.');
        
        const editor = vscode.window.activeTextEditor;
        if (!editor) return vscode.window.showWarningMessage('No active file to scan.');
        
        const filePath = editor.document.fileName;
        
        // Check if it's a binary file we can analyze
        if (!filePath.match(/\.(exe|dll|elf)$/i)) {
            return vscode.window.showWarningMessage('DAML: Active file is not a supported binary format (.exe, .dll, .elf)');
        }
        
        vscode.window.withProgress(
            { location: vscode.ProgressLocation.Notification, title: "DAML Scanning active file..." },
            async (progress) => {
                try {
                    progress.report({ message: 'Extracting features...' });
                    const features = await this.extractFeatures(filePath);
                    
                    progress.report({ message: 'Running model inference...' });
                    const result = await this.apiRequest('/predict', 'POST', { features });
                    
                    if (result.is_malicious) {
                        this.threatCount++;
                        this.recentDetections.unshift({
                            file: path.basename(filePath),
                            severity: result.confidence === 'high' ? 'critical' : result.confidence,
                            time: 'just now',
                            prob: result.malicious_probability
                        });
                        this.recentDetections = this.recentDetections.slice(0, 10);
                        
                        this.updateStatusBar();
                        this.updateWebview();
                        vscode.window.showWarningMessage(`DAML: Threat detected in ${path.basename(filePath)}! (${(result.malicious_probability * 100).toFixed(1)}% confidence)`);
                    } else {
                        vscode.window.showInformationMessage(`DAML: File appears safe (${(result.malicious_probability * 100).toFixed(1)}% malicious probability)`);
                    }
                } catch (err) {
                    vscode.window.showErrorMessage(`DAML: Scan failed - ${err}`);
                }
            }
        );
    }

    private async extractFeatures(filePath: string): Promise<number[]> {
        // Path to Python script in extension folder
        const pythonDir = path.join(this.context.extensionPath, 'python');
        const scriptPath = path.join(pythonDir, 'extract_features.py');
        
        try {
            // Run Python feature extraction
            const { stdout, stderr } = await execPromise(
                `python3 "${scriptPath}" "${filePath}"`,
                {
                    timeout: 30000,  // 30 second timeout
                    maxBuffer: 1024 * 1024 * 2  // 2MB buffer for large feature arrays
                }
            );
            
            if (stderr) {
                console.warn('Feature extraction stderr:', stderr);
            }
            
            // Parse JSON output
            const features = JSON.parse(stdout.trim());
            
            // Validate features array
            if (!Array.isArray(features)) {
                throw new Error('Feature extraction returned non-array');
            }
            
            if (features.length !== 2568) {
                throw new Error(`Expected 2568 features, got ${features.length}`);
            }
            
            // Validate all numbers are finite
            const invalidCount = features.filter((f: number) => !Number.isFinite(f)).length;
            if (invalidCount > 0) {
                throw new Error(`Feature extraction returned ${invalidCount} invalid values`);
            }
            
            return features;
            
        } catch (error: any) {
            console.error('Feature extraction failed:', error);
            
            // Provide specific error messages
            if (error.code === 'ENOENT') {
                throw new Error('Python3 not found. Please install Python 3.x');
            }
            if (error.killed) {
                throw new Error('Feature extraction timed out (30s limit)');
            }
            if (error.message?.includes('JSON')) {
                throw new Error('Feature extraction returned invalid data');
            }
            
            throw new Error(`Failed to extract features: ${error.message || error}`);
        }
    }

    private resolveThreats() {
        this.threatCount = 0;
        this.recentDetections = [];
        this.updateStatusBar();
        this.updateWebview();
        vscode.window.showInformationMessage('DAML: All threats resolved.');
    }

    private updateStatusBar() {
        this.statusBarItem.text = this.isEnabled 
            ? `$(shield) DAML ${this.apiAvailable ? '●' : '○'}` 
            : `$(circle-slash) DAML Off`;
        this.statusBarItem.tooltip = this.apiAvailable 
            ? `API Connected - ${this.threatCount} threats detected` 
            : 'API Disconnected';
        this.statusBarItem.backgroundColor = (this.isEnabled && this.threatCount > 0) 
            ? new vscode.ThemeColor('statusBarItem.warningBackground') 
            : undefined;
    }

    private updateWebview() {
        if (!this._view) return;
        
        if (!this.isEnabled) {
            this._view.webview.html = `<body style="display:flex;justify-content:center;align-items:center;height:100vh;opacity:0.5;">DAML is disabled</body>`;
            return;
        }

        const scriptUri = this._view.webview.asWebviewUri(
            vscode.Uri.file(path.join(this.context.extensionPath, 'media', 'dashboard.js'))
        );
        this._view.webview.html = this.getHtml(this._view.webview, scriptUri);
    }

    private getSeverityColor(severity: string): string {
        switch(severity) {
            case 'critical': return 'text-red-500';
            case 'high': return 'text-orange-500';
            case 'medium': return 'text-yellow-500';
            case 'low': return 'text-blue-400';
            default: return 'text-gray-400';
        }
    }

    private getSeverityIcon(severity: string): string {
        switch(severity) {
            case 'critical': return 'shield-alert';
            case 'high': return 'alert-triangle';
            case 'medium': return 'alert-circle';
            case 'low': return 'info';
            default: return 'file';
        }
    }

    private getHtml(webview: vscode.Webview, scriptUri: vscode.Uri) {
        const detectionsList = this.recentDetections.map(d => `
            <li class="p-2 flex items-center justify-between hover:bg-[var(--hover-bg)] cursor-pointer transition">
                <div class="flex items-center gap-2 overflow-hidden">
                    <i data-lucide="${this.getSeverityIcon(d.severity)}" class="w-3 h-3 ${this.getSeverityColor(d.severity)} flex-shrink-0"></i>
                    <div class="flex flex-col overflow-hidden">
                        <span class="truncate opacity-90">${d.file}</span>
                        <span class="text-[9px] opacity-50">${(d.prob * 100).toFixed(1)}% confidence</span>
                    </div>
                </div>
                <span class="opacity-50 flex-shrink-0 ml-2 text-[9px]">${d.time}</span>
            </li>
        `).join('');

        const emptyState = `
            <li class="p-4 text-center opacity-50 text-[10px]">
                <i data-lucide="shield-check" class="w-8 h-8 mx-auto mb-2 text-green-500"></i>
                No threats detected
            </li>
        `;

        return `<!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://unpkg.com/lucide@latest"></script>
            <style>
                :root {
                    --bg: var(--vscode-sideBar-background);
                    --fg: var(--vscode-foreground);
                    --border: var(--vscode-widget-border);
                    --accent: var(--vscode-textLink-foreground);
                    --card-bg: var(--vscode-editor-background);
                    --hover-bg: var(--vscode-list-hoverBackground);
                }
                body { 
                    background-color: var(--bg); 
                    color: var(--fg); 
                    font-family: var(--vscode-font-family); 
                    overflow-x: hidden; 
                    padding: 12px;
                }
                .card { 
                    background: var(--card-bg); 
                    border: 1px solid var(--border); 
                    border-radius: 6px; 
                }
                .chart-container { 
                    height: 160px; 
                    width: 100%; 
                    position: relative; 
                }
                ::-webkit-scrollbar { width: 6px; }
                ::-webkit-scrollbar-track { background: transparent; }
                ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
                .pulse {
                    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: .5; }
                }
            </style>
        </head>
        <body class="flex flex-col h-[100vh]">
            <div class="flex flex-col space-y-4 flex-none">
                
                <div class="flex justify-between items-center border-b border-[var(--border)] pb-2">
                    <div class="flex items-center gap-2">
                        <i data-lucide="shield" class="w-4 h-4 ${this.apiAvailable ? 'text-green-400' : 'text-red-400'} ${this.apiAvailable ? '' : 'pulse'}"></i>
                        <h1 class="text-xs font-bold uppercase tracking-wider opacity-80">DAML Status</h1>
                        ${!this.apiAvailable ? '<span class="text-[9px] text-red-400 ml-2">API Offline</span>' : ''}
                    </div>
                    <button id="clear-btn" class="bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded text-[10px] transition cursor-pointer ${this.threatCount === 0 ? 'opacity-50' : ''}" 
                            ${this.threatCount === 0 ? 'disabled' : ''}>
                        Clear Alerts
                    </button>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="alert-triangle" class="w-3 h-3 ${this.threatCount > 0 ? 'text-red-500' : 'text-green-500'}"></i> Threats
                        </p>
                        <p class="text-2xl font-bold ${this.threatCount > 0 ? 'text-red-500' : 'text-green-500'}">
                            ${this.threatCount}
                        </p>
                    </div>
                    <div class="card p-3 flex flex-col items-center justify-center text-center">
                        <p class="text-[10px] opacity-60 uppercase mb-1 flex items-center gap-1">
                            <i data-lucide="activity" class="w-3 h-3 ${this.apiAvailable ? 'text-green-500' : 'text-red-500'}"></i> API
                        </p>
                        <p class="text-2xl font-bold ${this.apiAvailable ? 'text-green-500' : 'text-red-500'}">
                            ${this.apiAvailable ? 'ON' : 'OFF'}
                        </p>
                    </div>
                </div>

                <div class="card p-2 chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>

                <div>
                    <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Quick Actions</h2>
                    <div class="grid grid-cols-2 gap-2">
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition ${!this.apiAvailable ? 'opacity-50' : ''}" 
                                onclick="vscode.postMessage({command: 'scanWorkspace'})" ${!this.apiAvailable ? 'disabled' : ''}>
                            <i data-lucide="folder-search" class="w-4 h-4 text-blue-400"></i>
                            <span>Scan Workspace</span>
                        </button>
                        <button class="card p-2 text-[10px] flex flex-col items-center justify-center gap-1 hover:bg-[var(--hover-bg)] cursor-pointer transition ${!this.apiAvailable ? 'opacity-50' : ''}" 
                                onclick="vscode.postMessage({command: 'scanActiveFile'})" ${!this.apiAvailable ? 'disabled' : ''}>
                            <i data-lucide="file-search" class="w-4 h-4 text-purple-400"></i>
                            <span>Scan Active File</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="mt-4 flex flex-col flex-grow overflow-hidden">
                <h2 class="text-[10px] font-bold uppercase opacity-60 mb-2">Recent Detections</h2>
                <div class="card p-0 overflow-y-auto flex-grow h-32">
                    <ul class="text-[10px] divide-y divide-[var(--border)]">
                        ${this.recentDetections.length > 0 ? detectionsList : emptyState}
                    </ul>
                </div>
            </div>

            <div class="mt-3 flex justify-between items-center text-[9px] opacity-50 border-t border-[var(--border)] pt-2 flex-none">
                <span>Engine v2.1.4</span>
                <span class="flex items-center gap-1">
                    <i data-lucide="${this.apiAvailable ? 'check-circle' : 'x-circle'}" class="w-3 h-3 ${this.apiAvailable ? 'text-green-500' : 'text-red-500'}"></i> 
                    ${this.apiAvailable ? 'Model Ready' : 'API Disconnected'}
                </span>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                
                document.getElementById('clear-btn')?.addEventListener('click', () => {
                    vscode.postMessage({command: 'resolveThreats'});
                });
                
                lucide.createIcons();
                
                // Chart.js for threat history
                const ctx = document.getElementById('weeklyChart').getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        datasets: [{
                            label: 'Threats',
                            data: [2, 1, ${this.threatCount}, 0, 1, 0, 0],
                            borderColor: 'rgb(239, 68, 68)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { display: false },
                            y: { display: false, min: 0 }
                        }
                    }
                });
            </script>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}